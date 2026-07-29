"""Probe: ROSVOT vs pyin on real da-da-da humming (HumTrans test split sample).

HumTrans (Tencent, CC-BY-NC-4.0) is EVALUATION-ONLY here: never use it to
train weights that ship. Ground truth is the INTENT midi -- the reference
melody subjects hummed along to -- so scores measure intent recovery, not
faithful transcription. Wavs are pulled selectively from the 14.7 GB zip via
HTTP ranged reads; nothing large lands on disk.

Run:  python eval/humtrans_probe.py [--n 20]
"""

import argparse
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))

DATA = os.path.join(HERE, "tmp", "humtrans")
WAV_URL = "https://huggingface.co/datasets/dadinghh2/HumTrans/resolve/main/all_wav.zip"
TOL_S = 0.1


def fetch_metadata():
    from huggingface_hub import hf_hub_download
    os.makedirs(DATA, exist_ok=True)
    keys_fn = hf_hub_download("dadinghh2/HumTrans", "train_valid_test_keys.json",
                              repo_type="dataset")
    midi_dir = os.path.join(DATA, "midi")
    if not os.path.isdir(midi_dir):
        midi_zip = hf_hub_download("dadinghh2/HumTrans", "all_midi.zip",
                                   repo_type="dataset")
        with zipfile.ZipFile(midi_zip) as z:
            z.extractall(midi_dir)
    return json.load(open(keys_fn, encoding="utf-8")), midi_dir


def fetch_wavs(test_keys, n):
    """Pull n test wavs out of the remote zip, one per distinct segment."""
    from remotezip import RemoteZip
    wav_dir = os.path.join(DATA, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    picked, seen = [], set()
    for key in test_keys:                      # personID_musicID_segmentID_rep
        seg = "_".join(key.split("_")[1:3])
        if seg in seen:
            continue
        seen.add(seg)
        picked.append(key)
        if len(picked) >= n:
            break
    missing = [k for k in picked
               if not os.path.exists(os.path.join(wav_dir, k + ".wav"))]
    if missing:
        with RemoteZip(WAV_URL) as z:
            names = {os.path.splitext(os.path.basename(m))[0]: m
                     for m in z.namelist() if m.endswith(".wav")}
            for k in missing:
                if k not in names:
                    print(f"warn: {k} not in zip")
                    continue
                with z.open(names[k]) as src, \
                        open(os.path.join(wav_dir, k + ".wav"), "wb") as dst:
                    dst.write(src.read())
                print(f"fetched {k}.wav")
    return [k for k in picked
            if os.path.exists(os.path.join(wav_dir, k + ".wav"))]


def find_midi(midi_dir, key):
    """Label midi may be named by full key or by musicID_segmentID."""
    cands = [key, "_".join(key.split("_")[1:3]), "_".join(key.split("_")[:3])]
    for root, _, files in os.walk(midi_dir):
        for fn in files:
            if fn.endswith((".mid", ".midi")) and os.path.splitext(fn)[0] in cands:
                return os.path.join(root, fn)
    return None


def load_gt(midi_fn):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_fn)
    gt = [(int(n.pitch), float(n.start))
          for inst in pm.instruments for n in inst.notes]
    gt.sort(key=lambda x: x[1])
    return gt


def match_f1(gt, pred, shift, chroma):
    used, hits = set(), 0
    for gp, go in gt:
        for i, (pp, po) in enumerate(pred):
            if i in used or abs(po + shift - go) > TOL_S:
                continue
            if (pp % 12 == gp % 12) if chroma else (pp == gp):
                used.add(i)
                hits += 1
                break
    prec = hits / len(pred) if pred else 0.0
    rec = hits / len(gt) if gt else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def score(gt, pred):
    """Best global time shift (recording latency), then exact/chroma F1."""
    if not pred:
        return 0.0, 0.0, 0.0
    shifts = np.arange(-2.0, 2.01, 0.02)
    best = max(shifts, key=lambda s: match_f1(gt, pred, s, chroma=True))
    return match_f1(gt, pred, best, False), match_f1(gt, pred, best, True), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    import music_perception_server as mps
    if not mps._rosvot_available():
        print("error: ROSVOT vendor missing -- run packaging/setup_rosvot.py")
        sys.exit(1)

    keys, midi_dir = fetch_metadata()
    test_keys = keys.get("TEST") or keys.get("test") or []
    print(f"test split size: {len(test_keys)}; sampling {args.n} distinct segments")
    picked = fetch_wavs(test_keys, args.n)

    rows = []
    for k in picked:
        wav = os.path.join(DATA, "wav", k + ".wav")
        midi_fn = find_midi(midi_dir, k)
        if not midi_fn:
            print(f"warn: no gt midi for {k}")
            continue
        gt = load_gt(midi_fn)
        row = {"key": k, "n_gt": len(gt)}
        for engine in ("rosvot", "pyin"):
            os.environ.pop("MPS_DISABLE_ROSVOT", None)
            if engine == "pyin":
                os.environ["MPS_DISABLE_ROSVOT"] = "1"
            r = mps.transcribe_melody(wav, bpm=120)
            pred = [(n["pitch"], n["start_beats"] * 0.5) for n in r["notes"]]
            f1, f1c, shift = score(gt, pred)
            row[engine] = {"f1": round(f1, 3), "f1_chroma": round(f1c, 3),
                           "n_pred": len(pred), "shift": round(float(shift), 2)}
        rows.append(row)
        print(f"{k}: gt={row['n_gt']:3d}  "
              f"rosvot F1={row['rosvot']['f1']:.2f}/{row['rosvot']['f1_chroma']:.2f} "
              f"({row['rosvot']['n_pred']})  "
              f"pyin F1={row['pyin']['f1']:.2f}/{row['pyin']['f1_chroma']:.2f} "
              f"({row['pyin']['n_pred']})")

    if rows:
        for eng in ("rosvot", "pyin"):
            f1 = np.mean([r[eng]["f1"] for r in rows])
            f1c = np.mean([r[eng]["f1_chroma"] for r in rows])
            print(f"MEAN {eng:7s} exact F1={f1:.3f}  chroma F1={f1c:.3f}")
        out = os.path.join(DATA, "probe_results.json")
        json.dump(rows, open(out, "w", encoding="utf-8"), indent=2)
        print("details:", out)


if __name__ == "__main__":
    main()
