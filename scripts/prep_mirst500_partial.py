"""Phase E item 2 (partial PoC): download a small subset of MIR-ST500.

Full MIR-ST500 is 500 YouTube URLs × ~30 MB = ~15 GB compressed audio +
hours of yt-dlp. This script grabs the first N songs as a feasibility
test for item 2's MIR-ST500 BiLSTM stack. The full dataset can be fetched
in a separate long-running session.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("yt-dlp missing"); sys.exit(1)

LINK_JSON = Path("/workspace/datasets/mirst500/repo/MIR-ST500_20210206/MIR-ST500_link.json")
OUT_DIR = Path("/workspace/datasets/mirst500/audio_partial")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LINK_JSON) as f:
        links = json.load(f)
    target = list(links.items())[: args.n]
    print(f"downloading {len(target)} songs to {OUT_DIR}")
    success = 0; failed = 0
    for song_id, url in target:
        out_path = OUT_DIR / f"{song_id}.mp3"
        if out_path.exists():
            print(f"  have {song_id}"); success += 1; continue
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(OUT_DIR / f"{song_id}.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio",
                                 "preferredcodec": "mp3",
                                 "preferredquality": "192"}],
            "quiet": True, "no_warnings": True,
            "socket_timeout": 30,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            success += 1
            if success % 5 == 0:
                print(f"  [{success}/{args.n}] downloaded {song_id}")
        except Exception as e:
            failed += 1
            print(f"  fail {song_id}: {str(e)[:80]}")
    print(f"DONE: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
