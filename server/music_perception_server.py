#!/usr/bin/env python3
"""
Music Perception MCP Server (stdio transport)
=============================================

A Model Context Protocol server that turns an audio file into *facts a text
LLM can act on*: integrated loudness, true peak, tempo, key, spectral balance
and clipping. It is the "ears" of a DAW-control agent -- the brain renders a
WAV (e.g. via reaper-mcp's render_to_wav) and calls a tool here to perceive it.

Design split (see README): this server does deterministic measurement only.
The numbers are exact and reproducible; it does NOT make subjective judgements
("muddy", "harsh") -- that is a separate, later, model-based tool.

Speaks newline-delimited JSON-RPC 2.0 on stdin/stdout, the same protocol as
reaper-mcp, so prism-core's mcp_client connects to it identically.

Dependencies (this server only -- the agent kernel stays zero-dependency):
  numpy, soundfile, pyloudnorm, librosa, scipy   (see requirements.txt)
All permissive licenses; pure pip, no external binary (no ffmpeg needed).
"""

import json
import os
import sys

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "music-perception-mcp"
SERVER_VERSION = "0.1.0"


# --------------------------------------------------------------------------
# Audio loading
# --------------------------------------------------------------------------
def _load(path):
    """Read an audio file to (samples, channels) float64 + sample rate."""
    if not isinstance(path, str) or not path:
        raise ValueError("path is required")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"audio file not found: {path}")
    import soundfile as sf  # lazy: keeps import errors close to the tool call
    data, rate = sf.read(path, always_2d=True, dtype="float64")
    return data, int(rate)


# --------------------------------------------------------------------------
# Loudness  (pyloudnorm integrated + scipy oversampled true peak + EBU LRA)
# --------------------------------------------------------------------------
def _loudness(data, rate):
    import numpy as np
    out = {"integrated_lufs": None, "loudness_range_lu": None,
           "true_peak_dbtp": None, "sample_peak_db": None}

    sp = float(np.max(np.abs(data))) if data.size else 0.0
    out["sample_peak_db"] = round(20 * np.log10(sp), 2) if sp > 0 else None

    # Integrated loudness (ITU-R BS.1770 / EBU R128), pure-python.
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(rate)
        d = data if data.shape[1] > 1 else data[:, 0]
        li = meter.integrated_loudness(d)
        out["integrated_lufs"] = round(float(li), 2) if np.isfinite(li) else None
    except Exception:
        pass

    # True peak: >=4x oversample then take the max -- catches inter-sample peaks
    # that the raw sample peak misses (BS.1770 true-peak method).
    try:
        from scipy.signal import resample_poly
        peak = 0.0
        for ch in range(data.shape[1]):
            up = resample_poly(data[:, ch], 4, 1)
            peak = max(peak, float(np.max(np.abs(up))))
        out["true_peak_dbtp"] = round(20 * np.log10(peak), 2) if peak > 0 else None
    except Exception:
        pass

    # Loudness range (EBU R128 / Tech 3342): gated P95-P10 of short-term loudness.
    try:
        out["loudness_range_lu"] = _loudness_range(data, rate)
    except Exception:
        out["loudness_range_lu"] = None

    return out


def _loudness_range(data, rate):
    """EBU R128 loudness range from 3s short-term windows (1s hop), gated.

    Reuses pyloudnorm's K-weighting filters; returns None if unavailable."""
    import numpy as np
    import pyloudnorm as pyln
    meter = pyln.Meter(rate)
    filters = getattr(meter, "_filters", None)
    if not filters:
        return None
    x = data.astype(np.float64)
    if x.ndim == 1:
        x = x[:, None]
    for f in filters.values():
        x = f.apply_filter(x)
    nch = x.shape[1]
    g = np.array([1.0, 1.0, 1.0, 1.41, 1.41])[:nch]
    win, hop = int(3.0 * rate), int(1.0 * rate)
    if win <= 0 or x.shape[0] < win:
        return None
    loud = []
    for s in range(0, x.shape[0] - win + 1, hop):
        seg = x[s:s + win]
        z = float(np.sum(g * np.mean(seg ** 2, axis=0)))
        if z > 0:
            loud.append(-0.691 + 10 * np.log10(z))
    loud = np.array([v for v in loud if v >= -70.0])  # absolute gate
    if loud.size < 2:
        return None
    rel = 10 * np.log10(np.mean(10 ** (loud / 10.0))) - 20.0  # relative gate
    kept = loud[loud >= rel]
    if kept.size < 2:
        return None
    return round(float(np.percentile(kept, 95) - np.percentile(kept, 10)), 2)


# --------------------------------------------------------------------------
# Clipping  (digital full-scale, pure numpy)
# --------------------------------------------------------------------------
def _clipping(data, rate, thr=0.999):
    import numpy as np
    peak = np.max(np.abs(data), axis=1) if data.ndim > 1 else np.abs(data)
    idx = np.where(peak >= thr)[0]
    regions = []
    if idx.size:
        splits = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate([[idx[0]], idx[splits + 1]])
        for s in starts[:20]:
            regions.append({"t": round(float(s) / rate, 3)})
    return {"clipped_samples": int(idx.size),
            "first_regions_seconds": regions,
            "threshold": thr,
            "note": "digital full-scale clipping only; inter-sample overs show "
                    "up in true_peak_dbtp instead"}


# --------------------------------------------------------------------------
# Tempo / key / spectral  (librosa)
# --------------------------------------------------------------------------
def _tempo(mono, rate, librosa, np):
    tempo, _ = librosa.beat.beat_track(y=mono, sr=rate)
    return {"bpm": round(float(np.atleast_1d(tempo)[0]), 1),
            "method": "librosa.beat_track",
            "note": "reliable for steady rhythms; unreliable for rubato / "
                    "free-time / ambient material"}


_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles.
_KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
           2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
           2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _key(mono, rate, librosa, np):
    chroma = librosa.feature.chroma_cqt(y=mono, sr=rate)
    cm = chroma.mean(axis=1)
    if cm.sum() <= 0:
        return {"key": None, "mode": None, "confidence": 0.0,
                "method": "krumhansl-chroma"}
    cm = cm / cm.sum()
    maj, minp = np.array(_KS_MAJ), np.array(_KS_MIN)

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / denom) if denom > 0 else 0.0

    best = (-2.0, None, None)
    for i in range(12):
        rot = np.roll(cm, -i)
        cmaj, cmin = corr(rot, maj), corr(rot, minp)
        if cmaj > best[0]:
            best = (cmaj, _PITCH_CLASSES[i], "major")
        if cmin > best[0]:
            best = (cmin, _PITCH_CLASSES[i], "minor")
    return {"key": best[1], "mode": best[2],
            "confidence": round((best[0] + 1) / 2, 2),
            "method": "krumhansl-chroma",
            "note": "heuristic, single global key; struggles with key changes "
                    "or weak/atonal tonality"}


def _spectral(mono, rate, librosa, np):
    spec = np.abs(librosa.stft(mono)) ** 2
    freqs = librosa.fft_frequencies(sr=rate)
    bands = {"sub": (20, 60), "bass": (60, 250), "low_mid": (250, 500),
             "mid": (500, 2000), "high_mid": (2000, 6000), "high": (6000, 20000)}
    total = float(spec.sum()) + 1e-12
    bands_db = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        bands_db[name] = round(10 * np.log10(float(spec[mask].sum()) / total + 1e-12), 1)
    centroid = float(librosa.feature.spectral_centroid(y=mono, sr=rate).mean())
    rolloff = float(librosa.feature.spectral_rolloff(y=mono, sr=rate).mean())
    return {"bands_db_rel": bands_db,
            "centroid_hz": round(centroid),
            "rolloff_hz": round(rolloff),
            "note": "band energy relative to total (dB), for reference-curve "
                    "comparison -- not an absolute/calibrated spectrum"}


# --------------------------------------------------------------------------
# Public tools
# --------------------------------------------------------------------------
def measure_loudness(path):
    """Lightweight: loudness only (no librosa import -- fast)."""
    data, rate = _load(path)
    return {"file": os.path.abspath(path),
            "duration_seconds": round(data.shape[0] / rate, 2),
            "sample_rate": rate, "channels": int(data.shape[1]),
            "loudness": _loudness(data, rate)}


def analyze_audio(path):
    """One-stop deterministic analysis of an audio file."""
    import numpy as np
    data, rate = _load(path)
    result = {
        "file": os.path.abspath(path),
        "duration_seconds": round(data.shape[0] / rate, 2),
        "sample_rate": rate,
        "channels": int(data.shape[1]),
        "loudness": _loudness(data, rate),
        "clipping": _clipping(data, rate),
    }
    mono = data.mean(axis=1).astype(np.float32)
    try:
        import librosa
    except ImportError:
        note = {"error": "librosa not installed (pip install -r requirements.txt)"}
        result["tempo"] = result["key"] = result["spectral"] = note
        return result
    result["tempo"] = _tempo(mono, rate, librosa, np)
    result["key"] = _key(mono, rate, librosa, np)
    result["spectral"] = _spectral(mono, rate, librosa, np)
    return result


# --------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------
TOOLS = []


def tool(name, description, schema, builder):
    TOOLS.append({"name": name, "description": description,
                  "inputSchema": schema, "_builder": builder})


def obj(props, required=None):
    return {"type": "object", "properties": props,
            "required": required or [], "additionalProperties": False}


tool(
    "analyze_audio",
    "Analyze an audio file (WAV or any libsndfile-readable format) and return "
    "exact, reproducible facts a mixing/mastering agent can act on: loudness "
    "(integrated LUFS, loudness range, true peak dBTP, sample peak), tempo "
    "(BPM), musical key, spectral balance (6 bands relative dB + centroid / "
    "rolloff), and digital clipping (count + first timestamps). Deterministic "
    "measurement only -- it does not give subjective 'sounds muddy' opinions. "
    "Pass an absolute path, e.g. one returned by reaper-mcp's render_to_wav.",
    obj({"path": {"type": "string"}}, ["path"]),
    lambda a: analyze_audio(a["path"]),
)

tool(
    "measure_loudness",
    "Fast loudness-only measurement of an audio file: integrated LUFS, "
    "loudness range (LU), true peak (dBTP) and sample peak (dB). Lighter than "
    "analyze_audio (skips tempo/key/spectral). Use for quick master-bus "
    "loudness checks against a target (e.g. -14 LUFS).",
    obj({"path": {"type": "string"}}, ["path"]),
    lambda a: measure_loudness(a["path"]),
)

TOOL_INDEX = {t["name"]: t for t in TOOLS}


# --------------------------------------------------------------------------
# JSON-RPC / MCP plumbing  (same shape as reaper-mcp)
# --------------------------------------------------------------------------
def make_result(rid, result):
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def make_error(rid, code, message):
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def handle_request(msg):
    method = msg.get("method")
    rid = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return make_result(rid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
    if method == "ping":
        return make_result(rid, {})
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "tools/list":
        return make_result(rid, {
            "tools": [{"name": t["name"], "description": t["description"],
                       "inputSchema": t["inputSchema"]} for t in TOOLS]
        })
    if method == "resources/list":
        return make_result(rid, {"resources": []})
    if method == "prompts/list":
        return make_result(rid, {"prompts": []})

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        spec = TOOL_INDEX.get(name)
        if not spec:
            return make_error(rid, -32602, f"unknown tool: {name}")
        try:
            ret = spec["_builder"](args)
            text = json.dumps(ret, ensure_ascii=False, indent=2)
            return make_result(rid, {"content": [{"type": "text", "text": text}],
                                     "isError": False})
        except ImportError as e:
            return make_result(rid, {
                "content": [{"type": "text", "text":
                             f"Missing dependency: {e}. Run "
                             f"`pip install -r requirements.txt`."}],
                "isError": True})
        except Exception as e:  # noqa: BLE001
            return make_result(rid, {
                "content": [{"type": "text", "text": f"Analysis error: {e}"}],
                "isError": True})

    if rid is None:
        return None
    return make_error(rid, -32601, f"method not found: {method}")


def main():
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8", newline="\n")
        except (AttributeError, ValueError):
            pass
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        try:
            response = handle_request(msg)
        except Exception as e:  # noqa: BLE001
            response = make_error(msg.get("id"), -32603, f"internal error: {e}")
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
