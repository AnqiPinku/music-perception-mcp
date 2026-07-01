#!/usr/bin/env python3
"""Freeze the music-perception MCP server into a self-contained onedir sidecar.

Run inside a CLEAN venv (see packaging/requirements-build.txt) so PyInstaller only
bundles the permissive audio stack, not torch/transformers/matplotlib:

    python -m venv venv
    venv/Scripts/pip install -r packaging/requirements-build.txt
    venv/Scripts/python packaging/build_sidecar.py

Output: dist/music_perception/  (onedir: music_perception.exe + _internal/).
The desktop app spawns that exe as an MCP stdio sidecar (protocol-identical to
`python music_perception_server.py`). ONEDIR (not onefile) so numba's on-disk JIT
cache is reused across launches (~2.4s cold vs ~16s recompile every onefile start).
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SERVER = os.path.join(REPO, "server", "music_perception_server.py")
DIST = os.path.join(REPO, "dist")
WORK = os.path.join(REPO, "build")

# Modules the deterministic tools never touch — keep them out of the bundle.
# NOTE: sklearn is a librosa optional dep; analyze_audio's code path (beat_track,
# chroma_cqt, stft, spectral_*) does not use it, so excluding it is safe AND saves
# ~15 MB. build_sidecar verifies analyze_audio still works after the freeze.
EXCLUDES = ["matplotlib", "tkinter", "IPython", "pytest", "sklearn",
            "numba.tests", "notebook"]


def main():
    if shutil.which("pyinstaller") is None and not _has_pyinstaller():
        sys.exit("PyInstaller not found — run inside the build venv "
                 "(pip install -r packaging/requirements-build.txt).")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir",
           "--name", "music_perception", "--console", "--log-level", "WARN",
           "--distpath", DIST, "--workpath", WORK, "--specpath", WORK]
    for e in EXCLUDES:
        cmd += ["--exclude-module", e]
    cmd.append(SERVER)
    print("[build] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    out = os.path.join(DIST, "music_perception")
    print("[build] onedir -> %s" % out)
    print("[build] size: %.0f MB" % (_dir_size(out) / 1e6))


def _has_pyinstaller():
    try:
        import PyInstaller  # noqa: F401
        return True
    except ImportError:
        return False


def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


if __name__ == "__main__":
    main()
