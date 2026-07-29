#!/usr/bin/env python3
"""Set up the vendored ROSVOT backend for transcribe_melody.

ROSVOT (Li et al., ACL 2024, MIT license) is a neural singing-voice
transcription model. transcribe_melody uses it as the primary backend when
vendor/ROSVOT/checkpoints exists, falling back to pyin otherwise.

This script makes the setup reproducible:
  1. clone https://github.com/RickyL-2000/ROSVOT into vendor/ROSVOT
  2. download checkpoints.zip from the authors' Google Drive and unzip
     (mirror this file into your own artifact store; Drive links can rot)
  3. patch two hardcoded-CUDA spots so CPU inference works

Python deps (beyond requirements.txt): torch, torchaudio (CPU builds are
fine), pretty_midi, pyworld, gdown, matplotlib, pyyaml, tqdm.

Run:  python packaging/setup_rosvot.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VENDOR = os.path.join(REPO, "vendor")
ROSVOT = os.path.join(VENDOR, "ROSVOT")
GDRIVE_ID = "1JNtNT37KiLq9uFQqHk7JFs-3trxd3bRh"   # checkpoints.zip (see README)

CPU_PATCHES = [
    # (file, old, new)
    (os.path.join("inference", "rosvot.py"),
     'device = torch.device(f"cuda:{int(rank)}")',
     'device = torch.device(f"cuda:{int(rank)}") if torch.cuda.is_available()'
     ' else torch.device("cpu")'),
    (os.path.join("inference", "rosvot.py"),
     "                batch = move_to_cuda(batch, int(rank))",
     "                if torch.cuda.is_available():\n"
     "                    batch = move_to_cuda(batch, int(rank))"),
]


def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def main():
    os.makedirs(VENDOR, exist_ok=True)
    if not os.path.exists(ROSVOT):
        run(["git", "clone", "--depth", "1",
             "https://github.com/RickyL-2000/ROSVOT.git", ROSVOT])
    ckpt = os.path.join(ROSVOT, "checkpoints", "rosvot", "model.pt")
    if not os.path.exists(ckpt):
        zip_path = os.path.join(ROSVOT, "checkpoints.zip")
        if not os.path.exists(zip_path):
            run([sys.executable, "-m", "gdown", GDRIVE_ID, "-O", zip_path])
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ROSVOT)
    for rel, old, new in CPU_PATCHES:
        fn = os.path.join(ROSVOT, rel)
        src = open(fn, encoding="utf-8").read()
        if new in src:
            continue
        if old not in src:
            print(f"warn: patch anchor not found in {rel}; upstream changed?")
            continue
        open(fn, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        print(f"patched {rel}")
    print("ok: vendor/ROSVOT ready" if os.path.exists(ckpt)
          else "error: checkpoints missing")


if __name__ == "__main__":
    main()
