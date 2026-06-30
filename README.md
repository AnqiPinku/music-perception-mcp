# music-perception-mcp

The **ears** of a DAW-control agent. An [MCP](https://modelcontextprotocol.io)
server that turns an audio file into *exact, reproducible facts a text LLM can
act on* — loudness, true peak, tempo, key, spectral balance, clipping.

```
 text-LLM brain (DeepSeek/…)  ── decides ──►  reaper-mcp.render_to_wav(...)  ──►  take.wav
        ▲                                                                            │
        └──────────────  facts (JSON)  ◄── music-perception-mcp.analyze_audio(take.wav)
```

The brain renders a WAV (e.g. via [reaper-mcp](https://github.com/AnqiPinku/reaper-mcp-v2)'s
`render_to_wav`), calls a tool here to *perceive* it, then decides the next
mixing action. This server is a **取数型 (data-fetch) MCP tool** in prism-core
terms: it returns context, it does not act on the DAW.

Speaks newline-delimited JSON-RPC 2.0 on stdin/stdout — the same protocol as
reaper-mcp, so prism-core's `mcp_client` connects to it identically.

## Scope: deterministic measurement only

This server measures. The numbers are exact and reproducible (same file →
same answer), computed by signal-processing libraries, **not** by an AI model.

It deliberately does **not** make subjective judgements — "sounds muddy",
"vocal is harsh", "the mood is sad". That perception is a *separate, later*
tool (`listen_subjective`, backed by an audio LLM such as Gemini) and lives
outside this v1 on purpose: the trustworthiness and use of "exact number" vs
"opinion" are different, so they are kept apart. See the music-agent design
docs for the two-layer plan.

## Tools

### `analyze_audio(path)`
One-stop analysis. Returns:

| Field | What you get | Library |
|---|---|---|
| `loudness.integrated_lufs` | Integrated loudness (ITU-R BS.1770 / EBU R128) | pyloudnorm |
| `loudness.loudness_range_lu` | Loudness range (dynamics), gated P95−P10 of short-term | pyloudnorm + numpy |
| `loudness.true_peak_dbtp` | True peak via 4× oversampling (catches inter-sample overs) | scipy |
| `loudness.sample_peak_db` | Raw sample peak | numpy |
| `tempo.bpm` | Estimated tempo | librosa |
| `key.key` / `key.mode` / `key.confidence` | Global musical key (Krumhansl-Schmuckler) | librosa |
| `spectral.bands_db_rel` | 6-band energy balance (sub/bass/low-mid/mid/high-mid/high), relative dB | librosa |
| `spectral.centroid_hz` / `rolloff_hz` | Brightness measures | librosa |
| `clipping` | Digital full-scale clip count + first timestamps | numpy |

### `measure_loudness(path)`
Loudness block only (integrated LUFS, range, true peak, sample peak). Skips
librosa, so it's fast — use it for quick master-bus checks against a target
(e.g. −14 LUFS for streaming).

Both take an **absolute path**, e.g. one returned by reaper-mcp's
`render_to_wav`. WAV is the expected input; any
[libsndfile](http://libsndfile.github.io/libsndfile/)-readable format works
(FLAC/OGG/AIFF). MP3/M4A are not guaranteed — render to WAV first.

## Capabilities and boundaries

What this server is good for — and where each number stops being trustworthy.
Read this before acting on a value.

| Metric | Reliable for | Boundary / caveat |
|---|---|---|
| **Integrated LUFS** | Master/stem loudness vs a target; A/B before-after | Whole-file integrated; not a live/streaming meter |
| **True peak (dBTP)** | Catching inter-sample overs before a limiter ceiling | 4× oversample (BS.1770 minimum); a hair below dedicated 8× meters but well within practical tolerance |
| **Loudness range (LU)** | Rough dynamics / over-compression check | EBU-style short-term implementation; treat as *indicative*, not certified |
| **Tempo (BPM)** | Steady electronic / pop / rock | Unreliable on rubato, free time, ambient, or no clear beat — returns **0.0** when it finds no beat (honest, not an error) |
| **Key** | Single-key tonal material | One **global** key only — misses modulations/key changes; weak on atonal/percussive/sparse audio; major-vs-minor can flip on ambiguous tonality. Use `confidence` |
| **Spectral bands** | Comparing a mix against a reference curve ("too much 2–6 kHz vs the reference") | **Relative** energy (dB vs total), not an absolute/calibrated spectrum; not loudness-weighted |
| **Clipping** | Detecting digital full-scale clipping | Full-scale only (≥0.999); soft/analog-style clipping and inter-sample overs are **not** here — those show up as a high `true_peak_dbtp` |

Cross-cutting:
- **Measurement, not opinion.** No "muddy/harsh/sad" — that's the future
  subjective layer.
- **Garbage in, garbage out.** Feed it the actual render. The numbers describe
  exactly the file you pass, including its sample rate and channel layout.
- **One global answer per file** for tempo/key. For per-section analysis,
  render that section (reaper-mcp `render_to_wav` with a time selection or
  `region:N`) and analyze it separately.

## Setup

```bash
pip install -r requirements.txt          # numpy soundfile pyloudnorm librosa scipy
python server/test_server.py             # offline self-test on a synthetic WAV
```

Register with an MCP client (e.g. prism-core / Claude Code) — add to your
`mcp_servers.json` / `.mcp.json`:

```json
{
  "mcpServers": {
    "music-perception": {
      "command": "python",
      "args": ["A:\\Prismcode\\music-perception-mcp\\server\\music_perception_server.py"]
    }
  }
}
```

## Dependencies & licensing

All dependencies are permissive (BSD/MIT/ISC) and pure-pip — **no external
binary, no ffmpeg**. They are confined to this server; the prism-core kernel
and the other MCP servers stay zero-dependency. Notably this avoids
`madmom` (non-commercial model weights) and `Essentia` (AGPL), so the stack
stays commercial-friendly.

## Roadmap (not in v1)

- `separate_stems(path)` — Demucs source separation (heavy; CPU-slow). Lets you
  measure each instrument's loudness/masking.
- `listen_subjective(path, question?)` — the subjective layer (audio LLM /
  Gemini): "does this sound muddy / harsh / what's the mood". Returns opinion
  JSON, kept separate from the exact numbers above.
