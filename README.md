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

The deterministic tools (`analyze_audio`, `measure_loudness`) make **no**
subjective judgement — no "muddy/harsh/sad". Those come from ONE clearly
separated, non-deterministic tool, `listen_subjective`, backed by an audio LLM
(Gemini). Exact numbers and opinions are kept apart on purpose — their
trustworthiness and use differ. An empirical benchmark backs this split:
deterministic MIR tracks controlled spectral defects perfectly (Spearman ρ≈1.0)
while models don't (≤0.31); models judge mood/emotion decently (0.46–0.64 vs
human DEAM ratings) while MIR is blind. See the music-agent design docs.

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

### `listen_subjective(path, question?)` — the one non-deterministic tool
Holistic "listening" judgement via an audio LLM (Gemini): 0-100
muddy/harsh/sibilant/bright, valence/arousal in [-10,10], a mood word,
timestamped issues, a one-line overall. Optional `question` focuses it
("is the vocal sibilant?"). Use it for mood / holistic feel; use `analyze_audio`
for exact numbers.

Needs a key — set env **before launching the server**:
- `GEMINI_API_KEY` — required.
- `GEMINI_BASE_URL` — optional; set it to use an **OpenAI-compatible relay**
  (e.g. PackyCode `https://www.packyapi.com/v1`, or OpenRouter). Unset → Google's
  native Gemini API.
- `GEMINI_MODEL` — default `gemini-2.5-flash` (use `gemini-2.5-flash-lite` to save).

Without a key it returns `{configured:false, error}` and the deterministic tools
keep working. Install a backend: `pip install openai` (relay) or `google-genai`
(native). It downsamples to 16 kHz mono, ≤20 s, before sending.

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
- **Measurement vs opinion.** The deterministic tools give exact numbers;
  `listen_subjective` gives the opinions ("muddy/harsh/sad") — separately, and
  non-deterministically.
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

## Roadmap

- `separate_stems(path)` — Demucs source separation (heavy; CPU-slow). Lets you
  measure each instrument's loudness/masking.
- (done) `listen_subjective` — the subjective/mood layer, see above.
