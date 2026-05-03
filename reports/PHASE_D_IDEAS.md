# Phase D — what to try after Phase B+2

Phase B+2 closed all six v2-spec items and a half-dozen Phase C extensions
(B62, B63, B64, B65, B66, B67, B68b, B69, B70, B71). The HumScribe v3.4
pipeline is at:

| dimension | metric | gap to ceiling | candidate Phase D fix |
|---|---|---|---|
| ASAP Romantic transcription (Liszt) | snap = 0.053 (B63 YMT3+) | ceiling 0.132 (B53 oracle); pipeline ceiling 0.132, real ceiling unknown | structural Romantic-specific DP variant; learned voice tracker |
| Vocadito offset20 (humming durations) | F1 = 0.439 (heuristic) | IAA 0.642, gap 20pp | learned duration head; offset-aware DP for humming |
| Vocadito noff (humming onsets) | F1 = 0.665 / **soft 0.6466** (heuristic) | IAA 0.740, gap 7.5pp | small but real headroom; learned approach needs much more data |
| MusicGen LoRA fine-tune | 2.36M trainable, smoke OK (B68b); loss-NaN | n/a | plumb delay-pattern in CE; curate (melody, arrangement) pairs |

## What we now know does NOT work

1. **Small learned voicing/onset detector on 40 Vocadito clips alone**
   (B10/B19/B42/B50/B52/B69) — every approach from BiLSTM through
   HuBERT/MERT plateaus at 0.45-0.62 F1, well below the heuristic 0.665.
   This is a data-quantity ceiling: 32 train clips per CV fold is below
   the threshold where any 100K+ parameter model can avoid memorisation.

2. **Pseudo-label augmentation alone** (B70, in flight as of this writing).
   Combining 40 Vocadito + 118 MTG-QBH pseudo-labels brings train set to
   158 clips, but the pseudo labels are the heuristic's own outputs —
   they teach the BiLSTM to imitate the heuristic, not to outperform it.

3. **Voicing entry hysteresis** (B47). Lowering the entry threshold lets
   in too much noise.

4. **Voicing exit hysteresis** (B62/item-4). Raises offset20 by < 1pp.

5. **HMM segmenter** (B4/B6/B45). HMM ceiling is below voicing baseline.

6. **End-to-end YourMT3+ on humming** (B66). YMT3+ trained on
   instrument tracks → noff_a1 = 0.497 vs heuristic 0.665 (-17pp).

## What the data points to as Phase D candidates

### High EV, well-scoped
1. **Romantic-specific DP variant for Liszt** — *the* unfixed
   structural weakness. Liszt oracle is 0.132 (B53/B54), so even
   perfect transcription wouldn't pass a normal snap metric. The fix
   is a different DP reward function for rubato / pedaled music.
   Possible direction: time-warping DP that allows beat-relative
   onsets to slip ±1 tatum cheaply.
2. **Offset-aware DP for humming** — directly attacks the offset20
   gap. Add a duration-snap penalty into the DP for humming-mode.
   B56 tried tempo-snap durations and it didn't help — but a *learned*
   duration head trained on Vocadito real labels might.
3. **MusicGen LoRA proper training** — once the delay-pattern
   plumbing is added (Phase D scope per B68b report), curate ~50
   (melody, arrangement) pairs and run a 500-2000 step fine-tune.
   B68b validates the LoRA path; the missing piece is the loss math.

### Medium EV, dataset-bound
4. **Combine Vocadito + MTG-QBH (real labels)** — the agent-time
   spec wanted MedleyDB-Melody (registration-gated). Hand-aligning
   ~5 MTG-QBH clips in MuseScore (~30 min) gives 5 real-label clips
   on top of the 40 Vocadito. Combined with vibrato-aware augmentation
   (random pitch shift ±2 semitones, time stretch 0.85-1.15×, vocal
   tract noise), this could push the BiLSTM into the heuristic's
   territory.
5. **Anticipatory Music Transformer (Thickstun et al. 2024)** as a
   demo flourish — accept transcribed MIDI prompt, generate a
   continuation. Bolt-on inference, no training. Adds "AI score
   continuation" to the demo.
6. **MERT or MusicFM features for a learned voice tracker** on
   Romantic ASAP. The greedy + adaptive_pj voice tracker hits a
   ceiling on dense chordal textures (Liszt 0.078). A small Transformer
   over (note onset, pitch, velocity, MERT context) → voice id
   classifier could close 30pp on hard pieces. Unlike the humming
   case, ASAP has thousands of MIDI scores with voice-track labels.

### Lower EV, demo-rich
7. **AudioLDM2 / MAGNeT** as MusicGen alternatives. Different
   generative audio architectures, useful for ablation but unlikely
   to outperform MusicGen-Melody-Large for melody following.
8. **Stereo arrangement output** via `facebook/musicgen-stereo-large`
   (text-only stereo, no melody). Bolt-on for richer-sounding demos
   when melody following isn't required.
9. **Beat-conditioned DP with learned slow-tempo beat tracker**.
   On real humming below 60 BPM, beat_this is the weakest link.
   Fine-tune the beat_this model on slow vocal content.

## Recommended Phase D order (highest EV first)

1. **MusicGen LoRA proper training** — completes B68b, tightens the
   "we trained a generative model" story.
2. **Romantic-specific DP for Liszt** — directly attacks the worst
   piece's failure mode.
3. **MERT-features Transformer voice tracker on ASAP** — addresses
   Liszt's voice-tracking failure (different angle than #2).
4. **Anticipatory Music Transformer demo** — additive demo flourish.
5. **Hand-aligned 5-clip MTG-QBH supplement** — push the learned
   voicing approach over the heuristic with a tractable amount of
   data work.

The first four are independent and parallelizable on the 32 GB Blackwell.
Five depends on a manual labeling step (MuseScore alignment).
