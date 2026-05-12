"""ME-9 — line-of-fifths enharmonic spelling (Temperley 2001).

For each transcribed note, choose the spelling (e.g. F# vs Gb) that minimises
the line-of-fifths distance to the surrounding context. Concretely: maintain
a running "tonal centre" that's a weighted average of recently-played notes'
line-of-fifths positions, and pick each new note's spelling to be closest to
that centre. When a key signature is known, prefer spellings inside the key.

Pure visual-quality polish — doesn't move F1, doesn't change durations,
doesn't change pitches. Just fixes accidentals so D# in E major stays D# and
not Eb. Cheap to run (CPU, O(N) over notes).

References:
- Temperley, D. (2001). The Cognition of Basic Musical Structures, MIT Press.
- Krumhansl, C. (1990). Cognitive Foundations of Musical Pitch, Oxford.

Implementation strategy:
1. Each MIDI pitch maps to a set of candidate spellings (e.g. MIDI 66 = F#4
   or Gb4). The candidates are positions on the line of fifths.
2. We slide a context window over the notes. For each note, score each
   candidate by (a) distance to the running tonal centre, (b) penalty for
   accidental mismatch with the key signature.
3. Pick the lowest-score spelling and update the running centre.

The result is applied by setting `note.pitch.step` and `note.pitch.accidental`
on the music21 notes before final musicxml export.
"""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass

from humscribe.notes import NoteEvent


# Line of fifths: ...Cb Gb Db Ab Eb Bb F C G D A E B F# C# G# D# A# E# B#...
# We index spellings by their position on this line. Lower index = flatter.
# Reference: C = 0.
_LOF_NAMES = ["F--", "C--", "G--", "D--", "A--", "E--", "B--",
              "F-", "C-", "G-", "D-", "A-", "E-", "B-",
              "F", "C", "G", "D", "A", "E", "B",
              "F#", "C#", "G#", "D#", "A#", "E#", "B#",
              "F##", "C##", "G##", "D##", "A##", "E##", "B##"]
_LOF_CENTRE = 15  # index of "C"

# Map line-of-fifths position -> pitch class (0..11)
_LOF_PC: list[int] = []
for n in _LOF_NAMES:
    base = "FCGDAEB".index(n[0])
    # Step semitones: F=5, C=0, G=7, D=2, A=9, E=4, B=11
    semis = [5, 0, 7, 2, 9, 4, 11][base]
    # Each '#' adds 1 semitone, each '-' subtracts 1.
    semis += n.count("#") - n.count("-")
    _LOF_PC.append(semis % 12)


def _candidates_for_pc(pc: int) -> list[int]:
    """Return all line-of-fifths indices that produce pitch class pc."""
    return [i for i, p in enumerate(_LOF_PC) if p == pc]


def _name_to_step_and_acc(name: str) -> tuple[str, int]:
    """('F#') -> ('F', +1); ('Bb') / ('B-') -> ('B', -1)."""
    letter = name[0]
    acc = name.count("#") - name.count("-") - name.count("b")
    return letter, acc


@dataclass
class _LOFConfig:
    """Tunable knobs. Defaults from Temperley 2001 chapter 3."""
    window_notes: int = 16
    centre_pull: float = 1.0       # distance-to-centre weight
    key_bonus: float = 3.5         # bonus for being in-key
    decay: float = 0.85            # how quickly old notes lose centre influence
    range_clip: int = 14           # cap candidates within ±14 of centre


def _key_to_lof(key_sharps: int | None, key_tonic_pc: int | None) -> int:
    """Estimated centre of key signature on the line of fifths.

    `key_sharps` (-7..+7) is sharpcount of the key signature.
    Returns the line-of-fifths index of the major-key tonic.
    """
    if key_sharps is not None:
        # 0 sharps = C major (LOF idx 15). Each sharp shifts +1 to the right.
        return _LOF_CENTRE + int(key_sharps)
    if key_tonic_pc is not None:
        # Default to major mode: tonic pc -> first LOF candidate within ±7.
        cands = _candidates_for_pc(int(key_tonic_pc) % 12)
        return min(cands, key=lambda i: abs(i - _LOF_CENTRE))
    return _LOF_CENTRE


def spell_notes_with_line_of_fifths(
    notes: Sequence[NoteEvent],
    key_sharps: int | None = None,
    key_tonic_pc: int | None = None,
    cfg: _LOFConfig | None = None,
) -> list[tuple[str, int]]:
    """Choose spellings for each note.

    Returns a list of (letter, alter) tuples, parallel to `notes`. `letter`
    is one of 'CDEFGAB'; `alter` is in {-2, -1, 0, +1, +2}.
    """
    cfg = cfg or _LOFConfig()
    centre_lof = _key_to_lof(key_sharps, key_tonic_pc)
    running = float(centre_lof)
    out: list[tuple[str, int]] = []
    for n in notes:
        midi = n.midi()
        if midi <= 0:
            out.append(("C", 0))
            continue
        pc = midi % 12
        cands = _candidates_for_pc(pc)
        # Restrict to a window around the current running centre — avoids
        # nonsensical triple-flat / triple-sharp spellings.
        cands = [c for c in cands if abs(c - centre_lof) <= cfg.range_clip]
        if not cands:
            cands = _candidates_for_pc(pc)
        scored = []
        for c in cands:
            dist = abs(c - running) * cfg.centre_pull
            key_dist = abs(c - centre_lof)
            in_key = 1.0 if key_dist <= 6 else 0.0
            score = dist - cfg.key_bonus * in_key
            scored.append((score, c))
        _, best = min(scored, key=lambda x: x[0])
        letter, alter = _name_to_step_and_acc(_LOF_NAMES[best])
        out.append((letter, alter))
        # Update running centre as a leaky integrator.
        running = cfg.decay * running + (1.0 - cfg.decay) * best
    return out


def spell_with_line_of_fifths(stream, key=None):
    """Walk a music21 Stream and re-spell every note via the line-of-fifths
    rule. `key` is an optional music21 Key (or KeySignature) used to seed
    the centre. Returns the stream in place (mutating note.pitch fields).
    """
    from music21 import note as m21note, chord as m21chord, key as m21key

    key_sharps: int | None = None
    key_tonic_pc: int | None = None
    if key is not None:
        if isinstance(key, m21key.Key):
            key_sharps = key.sharps
            key_tonic_pc = int(key.tonic.midi) % 12 if key.tonic else None
        elif isinstance(key, m21key.KeySignature):
            key_sharps = key.sharps

    notes: list[NoteEvent] = []
    refs: list = []  # parallel list of music21 elements per NoteEvent
    for el in stream.recurse().notes:
        if isinstance(el, m21note.Note):
            notes.append(NoteEvent(
                onset_s=0.0, offset_s=0.0,
                pitch_midi=int(el.pitch.midi), velocity=80,
            ))
            refs.append((el, None))
        elif isinstance(el, m21chord.Chord):
            for j, p in enumerate(el.pitches):
                notes.append(NoteEvent(
                    onset_s=0.0, offset_s=0.0,
                    pitch_midi=int(p.midi), velocity=80,
                ))
                refs.append((el, j))

    spellings = spell_notes_with_line_of_fifths(notes, key_sharps=key_sharps,
                                                  key_tonic_pc=key_tonic_pc)
    from music21 import pitch as m21pitch
    for (el, j), (letter, alter) in zip(refs, spellings):
        if j is None:
            old = el.pitch
            new = m21pitch.Pitch()
            new.step = letter
            new.octave = old.octave
            new.accidental = m21pitch.Accidental(alter)
            # Avoid changing actual pitch — verify the rebuilt pitch's MIDI
            # matches the original.
            if int(new.midi) != int(old.midi):
                continue
            el.pitch = new
        else:
            old = el.pitches[j]
            new = m21pitch.Pitch()
            new.step = letter
            new.octave = old.octave
            new.accidental = m21pitch.Accidental(alter)
            if int(new.midi) != int(old.midi):
                continue
            # Replace pitch j by name
            new_pitches = list(el.pitches)
            new_pitches[j] = new
            el.pitches = tuple(new_pitches)
    return stream
