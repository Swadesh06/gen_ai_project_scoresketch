"""MTG-QBH (Salamon et al. 2013) loader. mirdata 1.0.0 does not register this
dataset, so we download direct from Zenodo record 1290712 and expose the
mirdata-style API the eval scripts expect.

Layout after download:
    {data_home}/
      audio/             -> wav clips (the audio under various Zenodo file names)
      _MTG-QBH-extracted/ -> raw extracted contents we keep for provenance
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import tempfile
import urllib.request
import zipfile

ZENODO_RECORD = "1290712"
DEFAULT_DATA_HOME = Path("~/datasets/mtg_qbh").expanduser()
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
USER_AGENT = "humscribe/3.2 mtg_qbh-loader"


@dataclass
class MTGQBHTrack:
    track_id: str
    audio_path: str

    def __repr__(self) -> str:
        return f"MTGQBHTrack(track_id={self.track_id!r}, audio_path={self.audio_path!r})"


class MTGQBH:
    name = "mtg_qbh"

    def __init__(self, data_home: str | os.PathLike | None = None) -> None:
        self.data_home = Path(str(data_home)).expanduser() if data_home else DEFAULT_DATA_HOME
        self.audio_dir = self.data_home / "audio"
        self._extract_dir = self.data_home / "_MTG-QBH-extracted"

    def download(self, force: bool = False) -> None:
        self.data_home.mkdir(parents=True, exist_ok=True)
        if self.audio_dir.exists() and any(self.audio_dir.glob("*.wav")) and not force:
            return
        self._extract_dir.mkdir(parents=True, exist_ok=True)
        files = self._zenodo_files()
        if not files:
            raise RuntimeError(f"Zenodo record {ZENODO_RECORD} returned no files; check network")
        for f in files:
            self._download_one(f["links"]["self"], f["key"])
        self._materialize_audio()

    def validate(self) -> None:
        wavs = list(self.audio_dir.glob("*.wav")) if self.audio_dir.exists() else []
        if not wavs:
            raise RuntimeError(f"No wavs under {self.audio_dir}; run .download() first")

    def load_tracks(self) -> dict[str, MTGQBHTrack]:
        out: dict[str, MTGQBHTrack] = {}
        if not self.audio_dir.exists():
            return out
        for w in sorted(self.audio_dir.glob("*.wav")):
            tid = w.stem
            out[tid] = MTGQBHTrack(track_id=tid, audio_path=str(w))
        return out

    def _zenodo_files(self) -> list[dict]:
        req = urllib.request.Request(ZENODO_API, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            import json
            data = json.loads(r.read().decode("utf-8"))
        return data.get("files", [])

    def _download_one(self, url: str, key: str) -> Path:
        dest = self._extract_dir / key
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        if key.lower().endswith(".zip"):
            with zipfile.ZipFile(dest) as z:
                z.extractall(self._extract_dir)
        return dest

    def _materialize_audio(self) -> None:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        for wav in self._extract_dir.rglob("*.wav"):
            tgt = self.audio_dir / wav.name
            if not tgt.exists():
                shutil.copy2(wav, tgt)
