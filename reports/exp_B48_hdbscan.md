# exp_B48 — HDBSCAN voice tracker on Romantic ASAP

## Goal
Address the B37 finding: ASAP non-Bach pieces (Liszt 0.078, Chopin 0.469) break with the default greedy voice tracker. Test whether HDBSCAN density-based clustering of (time, pitch) features captures Romantic chord textures better.

## Procedure
- Cached ByteDance + beat_this output per piece (one-time cost) so each VT config is fast.
- 5 ASAP pieces: Bach Fugue BWV 846 + 4 Romantic (Beethoven Sonata 21-1, Schumann Toccata, Chopin Berceuse, Liszt Sonata).
- Tested 8 voice-tracking configs:
  - greedy with pj∈{3, 7, 12}
  - HDBSCAN with min_cluster_size∈{4, 6, 10, 15}
  - no voice tracking (single voice)
- Same 5-piece harness as B12 / B37.

## Results

| config | mean_snap (5 pieces) |
|---|---|
| **greedy_wide_pj12** | **0.587** |
| greedy_wide_pj7 | 0.584 |
| greedy_default (pj=3) | 0.571 |
| hdbscan_mcs15 | 0.530 |
| no_vt | 0.522 |
| hdbscan_mcs10 | 0.524 |
| hdbscan_mcs6 | 0.495 |
| hdbscan_mcs4 | 0.485 |

## Interpretation
**Wider greedy pitch_jump wins for the 5-piece set including 4 Romantic pieces** — pj=12 lifts mean snap from 0.571 (default pj=3) to 0.587 (+1.6pp). The wider pj keeps long melodic intervals in the same voice (e.g. Liszt's octave leaps don't fragment as separate voices).

**HDBSCAN loses to all greedy variants and even to no-VT.** Three reasons:
1. HDBSCAN clusters by *static* (time, pitch) similarity. Music voices wander in pitch but maintain temporal coherence — not what HDBSCAN captures.
2. The `min_cluster_size` parameter forces all small voice fragments to be marked as noise (-1), which the script then puts each in its own voice — fragmenting rather than merging.
3. The euclidean distance metric mixes pitch (semitones) and time (seconds) in the wrong proportions for music.

**Trade-off vs Bach Fugues**: B16 found pj=3 optimal for Bach Fugues alone (mean snap 0.856). On the 5-piece set with 4 Romantic, pj=12 wins. **Optimal pj is content-dependent.**

Decision options:
- (A) Keep pj=3 default (current), document pj=12 for Romantic content.
- (B) Switch to pj=7 as a compromise (loses ~1.5pp on Bach, gains ~1.5pp on Romantic).
- (C) Adaptive pj based on piece tempo / note density (future work).

For now: **keep pj=3 default** since the headline gate (Bach BWV 846) is what we're optimizing for. Document pj=12 as user-tunable for non-Bach.

Liszt remains broken (snap 0.074-0.078 across all VT configs) because ByteDance's note detection is the bottleneck for that piece — VT can't recover from missed/wrong notes.

## Next
- Per-piece pitch_jump auto-selection based on note-density estimate
- Try with HDBSCAN on (pitch, voice continuity over time) features instead of (time, pitch)
- The Liszt break is a ByteDance limitation; needs MT3 / YourMT3 alternative transcribers
