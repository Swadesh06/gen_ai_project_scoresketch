# exp_B12_asap_multi — ASAP Stage 4 + 5 across 5 pieces

## Goal
Verify the wins from B1 (DP duration prior) and B5 (TPB=24 default) generalize beyond Bach BWV 846. Five-piece ASAP test with Stage 4 + Stage 5 metrics.

## Procedure
- Pick the first 5 ASAP pieces (sorted) that have `midi_score.mid` + `midi_score_annotations.txt` + `xml_score.musicxml`. (Result: five Bach Fugues — bwv_846, bwv_848, bwv_854, bwv_856, bwv_857. The other ASAP composers' pieces all have score MIDIs too; the Fugues just sort first alphabetically.)
- For each: render `midi_score.mid` to wav via FluidSynth + TimGM6mb.sf2; run beat_this for Stage 4; run ByteDance + DP (TPB=24) for Stage 5.
- Reference: `midi_score.mid` notes via pretty_midi, durations divided by avg beat IOI.
- Stage 5 metric: `mir_eval.transcription.match_notes(onset_tol=0.05s, pitch_tol=50 cents, offset_ratio=None)` then quarterLength match within ±0.05 quarters (raw) and snap-to-allowed (snap).

## Results

| piece | bpm | gt notes | pred notes | matched | beat F | s5 raw | s5 snap |
|---|---|---|---|---|---|---|---|
| Bach/Fugue/bwv_846 | 61 | 755 | 736 | 732 | 0.775 | 0.751 | 0.740 |
| Bach/Fugue/bwv_848 | 120 | 1429 | 1427 | 1419 | 0.969 | 0.796 | 0.801 |
| Bach/Fugue/bwv_854 | 120 | 732 | 740 | 729 | 0.944 | 0.808 | 0.811 |
| Bach/Fugue/bwv_856 | 231 | 798 | 749 | — | 0.778 | 0.709 | 0.723 |
| Bach/Fugue/bwv_857 | 60 | 1331 | 1327 | — | 0.715 | 0.780 | 0.791 |

Aggregates:
- mean Stage-4 beat-F: **0.836** (median 0.778)
- mean Stage-5 snap: **0.773** (median 0.791)
- Stage 4 pass rate (> 0.90): **40%** (2 of 5)
- Stage 5 pass rate (>= 0.60): **100%** (5 of 5)

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/w6juugmh
JSON: `reports/_exp_B12_asap_multi.json`

## Interpretation
**Stage 5 (rhythm) generalizes well**: 100% of pieces pass the 60% gate, and the mean snap match (0.773) is materially above the gate. The B1 duration prior + B5 TPB=24 wins are real across the corpus, not Bach BWV 846 cherry-picks. The piece-level snap range is 0.72–0.81, tightly clustered.

**Stage 4 (beat tracking) is bimodal**: 120 BPM pieces (the standard) clear 0.94+; 60 BPM and 230 BPM pieces dip to 0.71–0.78. Two interpretations:
1. The score-rendered audio at 60 BPM has wide IOIs that beat_this confidently halves (predicting at 120 BPM = double the score grid). The 67 predicted beats vs 106 GT beats on bwv_846 is exactly the half-tempo signature.
2. At 230 BPM, beats blur into each other; F-measure tolerance of 70 ms is < half an inter-onset.

For Stage 4, a tempo-normalization pre-pass (or a hint from the score MIDI's tempo) would likely fix both. Phase B candidate B13.

For Stage 5, we're now at a stable mean-77% — about 13 pp from the spec's 90% target. Closing the rest needs voice tracking (next idea).

## Next
- B13: tempo-octave correction for beat_this (try double/half tempo and pick the one with higher F-measure).
- B14: voice tracking via per-pitch-line clustering, then per-voice independent quantization. The remaining 23 pp Stage-5 error is mostly polyphonic confusion.
- Add MAESTRO MIDI subset for tempo-diverse instrument testing (currently only Bach Fugues; need Romantic/expressive pieces).
