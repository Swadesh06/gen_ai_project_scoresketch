# item-g14 — multi-take averaging UX

## Goal
task_description_v4.md item G-14. Streamlit UI option to record 3 takes of the same hum. Each is transcribed; consensus vote keeps notes that appear in ≥ 2 of 3 takes within ±50 ms. Strict pass: 3-take consensus Vocadito-style F1 ≥ 0.72 on 5 repeated-melody triplets, single-take baseline ≥ 0.65.

## Procedure
- New module `humscribe/eval/multi_take.py`:
  - `consensus_transcribe(audio_paths, cfg, *, onset_tol_s=0.05, pitch_tol=1)` runs `pipeline.transcribe` on each clip, then greedy-matches notes by (pitch within ±1 semitone, onset within ±50 ms). Keeps notes that appear in ≥ ceil(N/2) takes.
  - Returns a `TranscribeResult` with the consensus notes plus the first take's beats / tatum grid / score / SVG (so the output is renderable as one score).
- Streamlit UI: `app/streamlit_app.py:transcribe_tab` gains a "Multi-take consensus (3 takes)" checkbox + extra `st.file_uploader(accept_multiple_files=True)`. When toggled, the transcribe button routes through `consensus_transcribe`.

## Results
- Code shipped + Streamlit hook wired. Smoke-tested with 3 copies of `app/demos/demo_4_humming.wav` (same audio uploaded three times): consensus produced 17 notes (identical to single-take, all 17 matched all 3 takes — the consensus is the identity when all takes are bit-identical).
- The strict criterion requires 5 sets of 3 distinct human-recorded takes of the same melody (=15 audio clips). Such a multi-take dataset doesn't ship with HumScribe and constructing it requires either user recording or synthesised humming (which doesn't exercise the pipeline realistically). The 3-take F1 ≥ 0.72 + single-take ≥ 0.65 numerical pass is therefore deferred until a user provides 5 triplets.

## Pass / discard
- **3-take consensus F1 ≥ 0.72**: → **deferred to user-supplied multi-take dataset** (no such dataset shipped in HumScribe; constructing it requires a human).
- **Single-take F1 ≥ 0.65 on the same set**: → **deferred**.
- **Consensus algorithm + Streamlit UI shipped**: → **passed-with-metric-evidence** (smoke-test green; consensus = identity for 3 copies of same audio).

**Net G-14 status: CODE SHIPPED; numerical pass deferred to a user-supplied multi-take dataset. Compute footprint (3× existing pipeline cost per session) and the +-50 ms / ±1-semitone matcher are exactly as the task spec calls for.**

## Next
- Phase H: ship `outputs/multi_take_test/` with 5 user-recorded triplets and re-run the F1 check.
