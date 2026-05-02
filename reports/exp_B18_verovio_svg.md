# exp_B18_verovio_svg — Verovio real-notation SVG rendering

## Goal
The Phase-A `humscribe.score.render_svg` fell back to a hand-rolled piano-roll SVG because LilyPond/MuseScore weren't on the pod. The piano-roll is informative but isn't proper notation — humans can't read rhythm, key, time signature, beaming, or articulation from it. Want real notation in WandB and on disk.

## Procedure
- `pip install verovio` (6.1.0). Pure Python — no external binaries.
- `humscribe/score.py:render_svg`: try Verovio first (`_verovio_svg(stream)`), then music21+LilyPond, then piano-roll. Verovio loads our MusicXML and `renderToSVG(1)` returns the first page's SVG.
- Verovio options set: `scale=40`, A4 page size, auto-fit page height, no header/footer.

## Results
- Vocadito clip 1: pipeline output SVG went from **5 KB piano-roll** to **121 KB engraved score**.
- Verovio emits warnings like "Insufficient space to draw mixed beam" for short clips with edge cases — these are layout hints, not errors. The SVG still renders.

Sample output: `outputs/demo_voc1_verovio.svg`. Open in any browser.

## Interpretation
Big qualitative win for human-facing output. Numerical metrics unchanged (Verovio doesn't touch the note quantization — it just renders the same MusicXML differently).

## Next
- Tune Verovio options (`pageWidth`, scale) per content density.
- Bump Vocadito gate's WandB run config to log Verovio SVGs as `wandb.Html` (currently piano-rolls).
- Add per-page SVGs for long pieces (currently only first page).
