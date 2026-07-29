"""Accuracy eval for transcribe_melody: known scores -> voice-like synthesis -> note-level F1.

Ground truth is a set of small melodies rendered as harmonic tones with
voice-like defects (vibrato, portamento, detuning, drift, rubato, breath noise,
legato). Each rendered clip is fed to the real transcribe_melody() and scored
note-by-note.

Two scores per case:
  faithful F1 -- prediction vs what was actually sung (nearest semitone of the
                 rendered pitch). Measures the TRANSCRIBER.
  intent F1   -- prediction vs the melody the singer intended (the written
                 score). The faithful-vs-intent gap is what a future
                 correction layer must close; the transcriber alone is not
                 expected to close it.

Run:  python eval/melody_eval.py            (from the repo root)
      python eval/melody_eval.py --quantize 0.25
Audio and JSON land in eval/tmp/ (gitignored).
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
import music_perception_server as mps  # noqa: E402

SR = 16000
ONSET_TOL_S = 0.05

# --------------------------------------------------------------------------
# Test melodies: (name, bpm, [(midi_pitch, start_beats, length_beats), ...])
# --------------------------------------------------------------------------
def _seq(pitches, dur, bpm, name):
    return (name, bpm, [(p, i * dur, dur) for i, p in enumerate(pitches)])

MELODIES = [
    _seq([60, 62, 64, 65, 67, 69, 71, 72], 1.0, 100, "scale_quarters"),
    _seq([60, 67, 64, 72, 55, 64, 60, 72], 0.5, 110, "leaps_eighths"),
    _seq([64, 64, 64, 62, 62, 60, 60, 60], 1.0, 100, "repeated_notes"),
    _seq([60, 62, 64, 62, 60, 64, 67, 64, 60, 62, 64, 62, 60, 57, 60, 60],
         0.25, 150, "fast_sixteenths"),
    _seq([45, 47, 48, 50, 48, 47, 45, 45], 1.0, 90, "low_register"),
    _seq([60, 64, 67, 72, 71, 67, 69, 76, 74, 72, 67, 64], 0.5, 120, "wide_tune"),
]

# --------------------------------------------------------------------------
# Voice-like synthesis
# --------------------------------------------------------------------------
def _render(melody, bpm, variant, rng):
    """Render note list -> (audio, sung_events). sung_events carry the
    actually-rendered onset (s) and the nearest-semitone pitch actually sung."""
    spb = 60.0 / bpm
    gap = 0.0 if variant == "legato" else 0.03
    total_s = (melody[-1][1] + melody[-1][2]) * spb + 0.5
    audio = np.zeros(int(total_s * SR), dtype=np.float64)
    events = []
    total_dur_s = (melody[-1][1] + melody[-1][2]) * spb
    prev_hz = None
    for pitch, start_b, len_b in melody:
        onset = start_b * spb
        if variant == "rubato":
            onset = max(0.0, onset + rng.uniform(-0.06, 0.06))
        dur = max(0.08, len_b * spb - gap)
        n = int(dur * SR)
        t = np.arange(n) / SR
        cents = 0.0
        if variant == "detuned":
            cents = 40.0
        elif variant == "drift":
            cents = 80.0 * ((onset + dur / 2) / total_dur_s)  # sharper over time
        f0 = 440.0 * 2 ** ((pitch - 69 + cents / 100.0) / 12.0)
        f0_t = np.full(n, f0)
        if variant == "vibrato":
            f0_t = f0 * 2 ** (0.6 / 12.0 * np.sin(2 * np.pi * 5.5 * t))
        if variant == "portamento" and prev_hz is not None:
            k = min(n, int(0.08 * SR))
            f0_t[:k] = np.geomspace(prev_hz, max(f0_t[k - 1], 1.0), k)
        phase = 2 * np.pi * np.cumsum(f0_t) / SR
        tone = np.zeros(n)
        for h, amp in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06)):
            tone += amp * np.sin(h * phase)
        if variant == "breathy":
            tone = 0.7 * tone + 0.12 * rng.standard_normal(n)
        env = np.ones(n)
        a, r = int(0.02 * SR), int(0.04 * SR)
        env[:a] = np.linspace(0, 1, a)
        env[-r:] *= np.linspace(1, 0, r)
        if variant == "breathy":
            env *= np.linspace(1.0, 0.55, n)
        s0 = int(onset * SR)
        seg = (tone * env)[: len(audio) - s0]
        audio[s0:s0 + len(seg)] += seg
        sung_pitch = int(round(pitch + cents / 100.0))
        events.append({"pitch": sung_pitch, "intent_pitch": pitch, "onset_s": onset})
        prev_hz = float(f0_t[-1])
    peak = np.abs(audio).max() or 1.0
    return (0.9 * audio / peak).astype(np.float32), events

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _match(truth, pred, chroma=False, key="pitch"):
    used, hits = set(), 0
    for gt in truth:
        for i, pr in enumerate(pred):
            if i in used or abs(pr["onset_s"] - gt["onset_s"]) > ONSET_TOL_S:
                continue
            same = (pr["pitch"] % 12 == gt[key] % 12) if chroma else (pr["pitch"] == gt[key])
            if same:
                used.add(i)
                hits += 1
                break
    prec = hits / len(pred) if pred else 0.0
    rec = hits / len(truth) if truth else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec, hits

def score_case(events, result, bpm):
    pred = [{"pitch": n["pitch"], "onset_s": n["start_beats"] * 60.0 / bpm}
            for n in result["notes"]]
    f1, prec, rec, hits = _match(events, pred)
    f1c, _, _, hits_c = _match(events, pred, chroma=True)
    f1i, _, _, _ = _match(events, pred, key="intent_pitch")
    return {"f1": round(f1, 3), "precision": round(prec, 3), "recall": round(rec, 3),
            "f1_chroma": round(f1c, 3), "f1_intent": round(f1i, 3),
            "octave_errors": hits_c - hits,
            "n_truth": len(events), "n_pred": len(pred)}

# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
VARIANTS = ["clean", "vibrato", "portamento", "detuned", "drift", "rubato",
            "breathy", "legato"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", type=float, default=0.0)
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "tmp")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(20260720)
    rows = []
    for name, bpm, melody in MELODIES:
        for variant in args.variants:
            audio, events = _render(melody, bpm, variant, rng)
            wav = os.path.join(out_dir, f"{name}__{variant}.wav")
            sf.write(wav, audio, SR)
            res = mps.transcribe_melody(wav, bpm=bpm, quantize_beats=args.quantize)
            row = {"melody": name, "variant": variant, **score_case(events, res, bpm)}
            rows.append(row)

    print(f"{'melody':<17}{'variant':<12}{'F1':>6}{'chromaF1':>10}{'intentF1':>10}"
          f"{'oct':>5}{'truth':>7}{'pred':>6}")
    for r in rows:
        print(f"{r['melody']:<17}{r['variant']:<12}{r['f1']:>6.2f}{r['f1_chroma']:>10.2f}"
              f"{r['f1_intent']:>10.2f}{r['octave_errors']:>5}{r['n_truth']:>7}{r['n_pred']:>6}")
    print("-" * 63)
    for variant in args.variants:
        vs = [r for r in rows if r["variant"] == variant]
        mean = lambda k: sum(r[k] for r in vs) / len(vs)  # noqa: E731
        print(f"{'MEAN':<17}{variant:<12}{mean('f1'):>6.2f}{mean('f1_chroma'):>10.2f}"
              f"{mean('f1_intent'):>10.2f}{sum(r['octave_errors'] for r in vs):>5}")
    overall = sum(r["f1"] for r in rows) / len(rows)
    print(f"\nOVERALL faithful F1: {overall:.3f}  ({len(rows)} cases)")
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"details: {os.path.join(out_dir, 'results.json')}")

if __name__ == "__main__":
    main()
