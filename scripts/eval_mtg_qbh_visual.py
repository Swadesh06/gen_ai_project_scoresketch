"""MTG-QBH qualitative reality check - generate SVGs for 10 well-known clips.
No quantitative gate; you visually inspect the produced scores against the
melodies you recognize. Run before demo day."""
import argparse
from pathlib import Path
import mirdata
from humscribe.pipeline import transcribe
from humscribe.config import PipelineConfig

KNOWN_MELODIES = [   # MTG-QBH track IDs you'll recognize from listening
    # (fill in after running `python -c "import mirdata; ..."` to list track IDs)
    # Pick clips whose target song you know - Twinkle, Yesterday, Frere Jacques, etc.
]

def main(mtg_dir, modes):
    d = mirdata.initialize("mtg_qbh", data_home=mtg_dir)
    tracks = d.load_tracks()
    chosen = KNOWN_MELODIES or list(tracks.keys())[:10]
    for mode in modes.split(","):
        out = Path(f"outputs/mtg_qbh_{mode}")
        out.mkdir(parents=True, exist_ok=True)
        for tid in chosen:
            tr = tracks[tid]
            cfg = PipelineConfig(input_kind="humming", mode=mode)
            r = transcribe(tr.audio_path, cfg)
            (out / f"{tid}.svg").write_text(r.svg)
            print(f"{tid}/{mode}:  notes={r.n_notes}  bpm={r.bpm:.1f}  out={out}/{tid}.svg")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtg-dir", default="~/datasets/mtg_qbh")
    ap.add_argument("--modes", default="soft,medium")
    main(**vars(ap.parse_args()))
