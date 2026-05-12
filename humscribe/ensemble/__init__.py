"""Phase E item 7 ensemble members (music-theory-guided correctors).

Each ME-N member lives in its own module and exposes a clean entry point
that the pipeline can wire in optionally:

    from humscribe.ensemble.me9_line_of_fifths import spell_with_line_of_fifths
    new_stream = spell_with_line_of_fifths(stream, key)

Members:
- ME-1  pYIN diversifier (planned)
- ME-4  Tonal-meter prior on DP (planned)
- ME-7  Anacrusis detection (planned)
- ME-9  Line-of-fifths enharmonic spelling
- ME-10 Meter-template ensemble (planned)
- ME-11 Formant-band onset detector (planned)
- ME-14 MV2H system-level ensemble selection (planned)
"""
from humscribe.ensemble.me1_pyin import track_pitch_pyin, vote_with_pesto_crepe
from humscribe.ensemble.me4_tonal_meter_prior import (
    load_or_build_prior,
    tonal_meter_log_prior,
)
from humscribe.ensemble.me7_anacrusis import (
    detect_anacrusis,
    shift_beats_for_pickup,
)
from humscribe.ensemble.me9_line_of_fifths import (
    spell_with_line_of_fifths,
    spell_notes_with_line_of_fifths,
)

__all__ = [
    "track_pitch_pyin", "vote_with_pesto_crepe",
    "load_or_build_prior", "tonal_meter_log_prior",
    "detect_anacrusis", "shift_beats_for_pickup",
    "spell_with_line_of_fifths", "spell_notes_with_line_of_fifths",
]
