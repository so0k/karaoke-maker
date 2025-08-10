"""
pip install ffmpeg-python assemblyai unidecode regex
# also install ffmpeg on your system (brew/apt/choco)

Usage:
  python align_song.py --audio ./audio/from_boilerplate_to_flow_vocals.mp3 --lyrics ./audio/from_boilerplate_to_flow_lyrics.txt --out ./audio/from_boilerplate_to_flow_timings.json
  (Use your isolated vocal stem if you have it; alignment is usually better.)
"""

import os, json, math, argparse, re, regex as rx
from unidecode import unidecode
import ffmpeg
import assemblyai as aai

# ---------- 1) Audio pre-processing (FFmpeg) ----------
def prep_audio(in_path, out_wav, target_sr=16000):
    """
    - converts to mono PCM WAV @ target_sr
    - applies EBU R128 loudness normalization (loudnorm)
    """
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    (
        ffmpeg
        .input(in_path)
        .output(
            out_wav,
            ac=1, ar=target_sr, f='wav',
            # gentle normalization; keep it deterministic for aligners
            af='loudnorm=I=-16:TP=-1.5:LRA=11'
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return out_wav, target_sr
# FFmpeg filter reference: loudnorm (EBU R128). :contentReference[oaicite:0]{index=0}

# ---------- 2) Load & parse your lyrics into blocks ----------
HEADING_RX = rx.compile(r'^\s*\[(verse|chorus|bridge|pre-chorus|intro|outro)\s*\d*\]\s*$', rx.I)

def parse_lyrics(path):
    """
    Accepts headings like:
      [Verse 1]
      [Chorus]
    Blank lines split lines logically.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()

    blocks, cur = [], {"type":"verse","lines":[]}
    for line in raw.splitlines():
        if HEADING_RX.match(line):
            if cur["lines"]:
                blocks.append(cur); cur = {"type":"verse","lines":[]}
            cur["type"] = HEADING_RX.match(line).group(1).lower()
            continue
        if line.strip() == "":
            # empty line -> line break inside current block
            cur["lines"].append([])
        else:
            # split into words; keep punctuation for display, but store a clean token for matching
            words = line.strip().split()
            if not cur["lines"] or cur["lines"][-1] != []:
                cur["lines"].append([])
            cur["lines"][-1].append(words)
    if cur["lines"]:
        blocks.append(cur)
    return blocks

# ---------- 3) Transcribe with word-level timestamps (AssemblyAI) ----------
def transcribe_words(wav_path, api_key):
    """Call AssemblyAI and return per-word timings in seconds plus duration.

    Also returns a serializable raw payload for caching.
    """
    aai.settings.api_key = api_key
    transcriber = aai.Transcriber()
    cfg = aai.TranscriptionConfig()
    tx = transcriber.transcribe(wav_path, cfg)
    words = [
        {"w": w.text, "start": (w.start or 0)/1000.0, "end": (w.end or 0)/1000.0}
        for w in (tx.words or [])
    ]
    raw = {
        "id": getattr(tx, "id", None),
        "audio_duration": getattr(tx, "audio_duration", None),
        "words": [
            {
                "text": getattr(w, "text", None),
                "start": getattr(w, "start", None),
                "end": getattr(w, "end", None),
                "confidence": getattr(w, "confidence", None),
            }
            for w in (tx.words or [])
        ],
    }
    return words, tx.audio_duration, raw


def save_transcript_cache(path, payload: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_transcript_cache(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Convert cached raw into aligner-friendly structure
    cached_words = [
        {"w": w.get("text", ""), "start": (w.get("start", 0) or 0)/1000.0, "end": (w.get("end", 0) or 0)/1000.0}
        for w in (raw.get("words") or [])
    ]
    duration = raw.get("audio_duration")
    return cached_words, duration, raw
# AssemblyAI Python SDK returns per-word start/end; typical usage as above. :contentReference[oaicite:1]{index=1}

# ---------- 4) Token matching: map ASR words -> your lyrics exactly ----------
def norm_token(s):
    # Remove punctuation, lower, ASCII fold
    s = unidecode(s).lower()
    s = rx.sub(r"[^\p{Letter}\p{Number}']+", "", s)
    return s

def levenshtein(a: str, b: str) -> int:
    # Simple DP edit distance (insert/delete/substitute cost 1)
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, cur = cur, prev
    return prev[lb]


def token_similarity(a: str, b: str) -> float:
    a = norm_token(a)
    b = norm_token(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    d = levenshtein(a, b)
    m = max(len(a), len(b))
    return 1.0 - (d / m)

def align_words_to_lyrics(lyrics_blocks, asr_words):
    # Prepare a flat list of normalized ASR tokens
    asr = [{"i": i, "w": w["w"], "n": norm_token(w["w"]), "t": w["start"], "t_end": w["end"]} for i, w in enumerate(asr_words)]
    j = 0
    N = len(asr)
    out_blocks = []
    cur_time = 0.0

    # Small alias map for common ASR confusions (domain-specific acronyms etc.)
    ALIASES = {
        "hcl": {"htl", "hcl"},
        "htl": {"hcl", "htl"},
    }

    def find_best_match(token_norm: str, start_idx: int, window: int = 40, max_time: float | None = None, expected_time: float | None = None):
        # Search forward in a limited window for a close-enough match, constrained by time
        best_k, best_sim = None, -1.0
        best_score = -1e9
        end_idx = min(N, start_idx + window)
        for k in range(start_idx, end_idx):
            cand = asr[k]["n"]
            if not cand:
                continue
            # hard time cutoff to avoid matching far-ahead duplicates within a line
            if max_time is not None and asr[k]["t"] > max_time:
                break
            # Boost if alias-equivalent
            if token_norm in ALIASES and cand in ALIASES[token_norm]:
                sim = 1.0
            else:
                sim = token_similarity(token_norm, cand)
            # Avoid matching ultra-short function words fuzzily
            if len(token_norm) <= 2 and sim < 1.0:
                continue
            # Prefer closer-in-time candidates around expected_time
            time_bonus = 0.0
            if expected_time is not None:
                dt = abs(asr[k]["t"] - expected_time)
                # small penalty per second away from expected time
                time_bonus = -0.15 * dt
            score = sim + time_bonus
            if score > best_score or (score == best_score and sim > best_sim):
                best_k, best_sim, best_score = k, sim, score
            # short-circuit perfect match
            if best_sim == 1.0:
                break
        return best_k, best_sim

    last_asr_end = asr[-1]["t_end"] if asr else None
    for b in lyrics_blocks:
        out_b = {"type": b["type"], "start": None, "end": None, "lines": []}
        for line_group in b["lines"]:
            # line_group is list of word-arrays that were separated by blank lines during parsing
            for words_in_line in line_group:
                tokens = [{"w": w, "n": norm_token(w)} for w in words_in_line]
                line_words = []
                first_real_idx = None
                last_real_t = cur_time

                # Pass 0: choose an early, distinctive anchor to avoid skipping earlier content
                anchor_line_idx = None
                anchor_asr_k = None
                anchor_sim = -1.0
                anchor_time = None
                base_j = j
                for li, tok in enumerate(tokens):
                    n = tok["n"]
                    if len(n) <= 2:
                        continue
                    k, sim = find_best_match(n, base_j)
                    if k is None:
                        continue
                    # Anchor acceptance thresholds
                    L = len(n)
                    thr = 0.66 if L == 3 else 0.75
                    if n in ALIASES:
                        sim = 1.0
                    if sim >= thr:
                        t = asr[k]["t"]
                        if (anchor_time is None) or (t < anchor_time) or (t == anchor_time and sim > anchor_sim):
                            anchor_line_idx, anchor_asr_k, anchor_sim, anchor_time = li, k, sim, t

                # If we found a suitable anchor, set j to it and pre-fill placeholders up to anchor
                if anchor_line_idx is not None:
                    # pre-fill earlier tokens as unmatched for later backfill
                    for _ in range(anchor_line_idx):
                        fallback_t = (last_real_t or cur_time) + 0.20
                        line_words.append({"w": tokens[len(line_words)]["w"], "t": fallback_t, "_matched": False})
                        last_real_t = fallback_t
                    # Add the anchor token as matched
                    t = asr[anchor_asr_k]["t"]
                    line_words.append({"w": tokens[anchor_line_idx]["w"], "t": t, "_matched": True})
                    first_real_idx = len(line_words) - 1
                    last_real_t = asr[anchor_asr_k]["t_end"]
                    j = anchor_asr_k + 1

                for idx, tok in enumerate(tokens):
                    # skip tokens already placed (prior to anchor and the anchor itself)
                    if anchor_line_idx is not None and idx <= anchor_line_idx:
                        continue
                    # skip empty tokens
                    if tok["n"] == "":
                        continue
                    # Drop leading empty ASR tokens
                    while j < N and asr[j]["n"] == "":
                        j += 1
                    # Find best forward fuzzy match
                    exp_t = (last_real_t or cur_time) + 0.3
                    k, sim = find_best_match(tok["n"], j, max_time=(last_real_t + 3.0) if last_real_t else None, expected_time=exp_t)
                    accept = False
                    if k is not None:
                        # similarity threshold by length; len>=4 allow small edit, len 3 require >=0.67, <=2 exact only
                        L = len(tok["n"]) 
                        thr = 1.0 if L <= 2 else (0.66 if L == 3 else 0.75)
                        if sim >= thr:
                            accept = True
                    if accept:
                        t = asr[k]["t"]
                        line_words.append({"w": tok["w"], "t": t, "_matched": True})
                        first_real_idx = first_real_idx if first_real_idx is not None else len(line_words) - 1
                        last_real_t = asr[k]["t_end"]
                        j = k + 1
                    else:
                        # no match: temporarily mark as missing; we'll backfill later
                        fallback_t = (last_real_t or cur_time) + 0.20
                        line_words.append({"w": tok["w"], "t": fallback_t, "_matched": False})
                        last_real_t = fallback_t

                # Backfill missing timings: distribute unmatched words around the nearest matched anchors
                # Find indices of matched words
                matched_indices = [i for i, w in enumerate(line_words) if w.get("_matched")]
                if matched_indices:
                    # Leading unmatched (before first matched anchor)
                    first_idx = matched_indices[0]
                    first_t = line_words[first_idx]["t"]
                    # Space leading words backwards before first_t
                    gap = 0.24  # slightly larger to better cover missed onsets
                    for i in range(first_idx - 1, -1, -1):
                        prev_t = line_words[i + 1]["t"]
                        t = max(prev_t - gap, cur_time + 0.05)
                        line_words[i]["t"] = t
                    # Internal gaps between matched anchors -> linear interpo
                    prev_anchor = matched_indices[0]
                    for anchor in matched_indices[1:]:
                        t0 = line_words[prev_anchor]["t"]
                        t1 = line_words[anchor]["t"]
                        span = max(0.01, t1 - t0)
                        n_between = anchor - prev_anchor - 1
                        if n_between > 0:
                            step = span / (n_between + 1)
                            for k in range(1, n_between + 1):
                                line_words[prev_anchor + k]["t"] = t0 + step * k
                        prev_anchor = anchor
                    # Trailing unmatched after last anchor -> step forward with small gaps
                    last_anchor = matched_indices[-1]
                    last_t = line_words[last_anchor]["t"]
                    for i in range(last_anchor + 1, len(line_words)):
                        last_t += 0.20
                        line_words[i]["t"] = last_t
                else:
                    # No matches at all: keep increasing from cur_time
                    t = max(cur_time + 0.05, line_words[0]["t"] if line_words else cur_time)
                    for i in range(len(line_words)):
                        if i == 0:
                            line_words[i]["t"] = t
                        else:
                            t += 0.20
                            line_words[i]["t"] = t

                # Clean helper keys and compute start/end; clamp to last ASR time + small margin
                clamp_to = (last_asr_end + 0.5) if last_asr_end is not None else None
                for w in line_words:
                    if "_matched" in w:
                        del w["_matched"]
                    if clamp_to is not None and w["t"] > clamp_to:
                        w["t"] = clamp_to
                line_start = line_words[0]["t"] if line_words else cur_time
                line_end = line_words[-1]["t"] if line_words else cur_time
                cur_time = max(cur_time, line_end)

                out_b["lines"].append({"start": line_start, "end": line_end, "words": line_words})
                out_b["start"] = out_b["start"] if out_b["start"] is not None else line_start
                out_b["end"] = line_end
        out_blocks.append(out_b)
    return out_blocks

def build_output(blocks, sample_rate, source, duration):
    return {
        "blocks": blocks,
        "meta": {
            "sample_rate": sample_rate,
            "channel_layout": "mono",
            "source": source,
            "duration": duration
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to mp3 or wav (use vocal stem if available)")
    ap.add_argument("--lyrics", required=True, help="Path to lyrics.txt with [Verse]/[Chorus] headings")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--aai_key", default=os.getenv("ASSEMBLYAI_API_KEY"))
    ap.add_argument("--cache", default=None, help="Path to save/load raw transcript JSON (defaults to <audio>_transcript.json)")
    ap.add_argument("--reuse_cache", action="store_true", help="Reuse transcript cache if it exists to skip API")
    args = ap.parse_args()

    # Allow running without API key when reusing cache
    if not args.aai_key and not args.reuse_cache:
        raise SystemExit("Set --aai_key or ASSEMBLYAI_API_KEY, or pass --reuse_cache with an existing --cache file")

    # 1) Preprocess audio for robust alignment
    out_wav = os.path.join(os.path.dirname(args.out) or ".", "prep.wav")
    if os.path.exists(out_wav):
        wav_path = out_wav
        try:
            probe = ffmpeg.probe(wav_path)
            sr = int(next(s for s in probe["streams"] if s["codec_type"] == "audio")["sample_rate"])
        except Exception:
            sr = None
    else:
        wav_path, sr = prep_audio(args.audio, out_wav)

    # 2) Parse lyrics into blocks/lines
    lyrics_blocks = parse_lyrics(args.lyrics)

    # 3) Upload+transcribe -> per-word times
    cache_path = (
        args.cache
        if args.cache is not None
        else os.path.splitext(args.audio)[0] + "_transcript.json"
    )
    if args.reuse_cache and os.path.exists(cache_path):
        words, dur, raw = load_transcript_cache(cache_path)
        source = f"assemblyai_cache:{os.path.basename(cache_path)}"
    else:
        if not args.aai_key:
            raise SystemExit("--reuse_cache specified but cache not found; also no API key for live transcription")
        words, dur, raw = transcribe_words(wav_path, args.aai_key)
        save_transcript_cache(cache_path, raw)
        source = "assemblyai"

    # 4) Map ASR tokens back to your exact lyrics text
    blocks = align_words_to_lyrics(lyrics_blocks, words)

    # 5) Save timings.json
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(build_output(blocks, sr, source, dur), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
