#!/usr/bin/env python3
"""
Offline test for music_perception_server.py.

Generates a synthetic stereo WAV (a tone that clips at full scale), spawns the
server as a subprocess, talks MCP JSON-RPC over stdio, and checks that real
analysis comes back with sane values. Requires the analysis deps installed
(numpy, soundfile, pyloudnorm, librosa, scipy).

Run:  python test_server.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "music_perception_server.py")


def make_wav(path, seconds=3.0, rate=48000, freq=220.0):
    """A 220 Hz (~A3) sine driven past full scale so it clips -- gives us a
    known key region (A), measurable loudness, and guaranteed clipping."""
    import numpy as np
    import soundfile as sf
    t = np.arange(int(seconds * rate)) / rate
    sig = 1.2 * np.sin(2 * np.pi * freq * t)      # 1.2 > 1.0 -> will clip
    sig = np.clip(sig, -1.0, 1.0).astype(np.float32)
    stereo = np.stack([sig, sig], axis=1)
    sf.write(path, stereo, rate, subtype="PCM_24")


def make_burst_wav(path, rate=48000):
    """12s 音频：安静段(0-6s, -23dB) + 响段(6-9s, -6dB) + 安静段(9-12s)。
    用于验证 short-term 时间序列能定位"哪几秒过响"。"""
    import numpy as np
    import soundfile as sf
    t = np.arange(int(12.0 * rate)) / rate
    sig = 0.07 * np.sin(2 * np.pi * 220.0 * t)
    burst = (t >= 6.0) & (t < 9.0)
    sig[burst] = 0.5 * np.sin(2 * np.pi * 220.0 * t[burst])
    stereo = np.stack([sig, sig], axis=1).astype(np.float32)
    sf.write(path, stereo, rate, subtype="PCM_24")


def rpc(proc, msg):
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    if msg.get("id") is None:
        return None
    return json.loads(proc.stdout.readline())


def main():
    wav = os.path.join(tempfile.mkdtemp(prefix="mp_"), "tone.wav")
    make_wav(wav)

    proc = subprocess.Popen(
        [sys.executable, SERVER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
        text=True, encoding="utf-8",
        env=dict(os.environ, GEMINI_API_KEY=""),   # keep listen_subjective unconfigured
    )
    failures = []

    def check(label, cond, extra=""):
        print(("  PASS" if cond else "  FAIL"), label, extra)
        if not cond:
            failures.append(label)

    try:
        r = rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        check("initialize returns serverInfo",
              r["result"]["serverInfo"]["name"] == "music-perception-mcp")

        rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        r = rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in r["result"]["tools"]}
        check("tools/list has analyze_audio + measure_loudness + listen_subjective",
              {"analyze_audio", "measure_loudness", "listen_subjective"} <= names)

        # measure_loudness
        r = rpc(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "measure_loudness",
                                  "arguments": {"path": wav}}})
        body = json.loads(r["result"]["content"][0]["text"])
        check("measure_loudness: sample rate + channels",
              body["sample_rate"] == 48000 and body["channels"] == 2)
        ld = body["loudness"]
        check("measure_loudness: integrated LUFS is a finite number",
              isinstance(ld["integrated_lufs"], (int, float)),
              f"(LUFS={ld['integrated_lufs']})")
        check("measure_loudness: true peak near 0 dBTP for full-scale signal",
              isinstance(ld["true_peak_dbtp"], (int, float))
              and ld["true_peak_dbtp"] > -3.0,
              f"(TP={ld['true_peak_dbtp']})")

        # measure_loudness: BS.1770 三时间尺度（burst 定位）
        burst = os.path.join(os.path.dirname(wav), "burst.wav")
        make_burst_wav(burst)
        r = rpc(proc, {"jsonrpc": "2.0", "id": 31, "method": "tools/call",
                       "params": {"name": "measure_loudness",
                                  "arguments": {"path": burst}}})
        ld = json.loads(r["result"]["content"][0]["text"])["loudness"]
        st = ld.get("short_term_series") or {}
        points = st.get("points") or []
        check("loudness scales: short_term_max >= integrated",
              isinstance(ld["short_term_max_lufs"], (int, float))
              and ld["short_term_max_lufs"] >= ld["integrated_lufs"] - 0.5,
              f"(st={ld['short_term_max_lufs']} int={ld['integrated_lufs']})")
        check("loudness scales: momentary_max >= short_term_max",
              isinstance(ld["momentary_max_lufs"], (int, float))
              and ld["momentary_max_lufs"] >= ld["short_term_max_lufs"] - 0.5,
              f"(mom={ld['momentary_max_lufs']})")
        check("loudness scales: series covers the file",
              len(points) >= 8 and st.get("window_s") == 3.0,
              f"(points={len(points)})")
        peak_t = max(points, key=lambda p: p[1])[0] if points else -1
        check("loudness scales: series localizes the 6-9s burst",
              5.0 <= peak_t <= 7.0, f"(peak window starts at {peak_t}s)")

        # analyze_audio
        r = rpc(proc, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                       "params": {"name": "analyze_audio",
                                  "arguments": {"path": wav}}})
        body = json.loads(r["result"]["content"][0]["text"])
        check("analyze_audio: duration ~3.0s",
              abs(body["duration_seconds"] - 3.0) < 0.2,
              f"(dur={body['duration_seconds']})")
        check("analyze_audio: detects clipping",
              body["clipping"]["clipped_samples"] > 0,
              f"(clipped={body['clipping']['clipped_samples']})")
        check("analyze_audio: tempo bpm present",
              isinstance(body["tempo"].get("bpm"), (int, float)),
              f"(bpm={body['tempo'].get('bpm')})")
        check("analyze_audio: key in the 12 pitch classes",
              body["key"]["key"] in
              ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
              f"(key={body['key'].get('key')} {body['key'].get('mode')})")
        check("analyze_audio: 6 spectral bands",
              len(body["spectral"]["bands_db_rel"]) == 6)

        # bad path -> isError
        r = rpc(proc, {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                       "params": {"name": "analyze_audio",
                                  "arguments": {"path": "Z:/nope/missing.wav"}}})
        check("missing file -> isError", r["result"]["isError"] is True)

        r = rpc(proc, {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                       "params": {"name": "nope", "arguments": {}}})
        check("unknown tool errors", "error" in r)

        # listen_subjective degrades gracefully without a key (no network)
        r = rpc(proc, {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                       "params": {"name": "listen_subjective",
                                  "arguments": {"path": wav}}})
        body = json.loads(r["result"]["content"][0]["text"])
        check("listen_subjective without key -> configured:false",
              body.get("configured") is False)

    finally:
        proc.stdin.close()
        proc.terminate()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):", failures)
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    main()
