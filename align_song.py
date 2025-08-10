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
    aai.settings.api_key = api_key
    transcriber = aai.Transcriber()
    # enable punctuation/formatting; we mainly need per-word times
    cfg = aai.TranscriptionConfig()
    tx = transcriber.transcribe(wav_path, cfg)
    # each word has .start/.end (ms) and .text
    words = [{"w": w.text, "start": w.start/1000.0, "end": w.end/1000.0} for w in tx.words or []]
    return words, tx.audio_duration
# AssemblyAI Python SDK returns per-word start/end; typical usage as above. :contentReference[oaicite:1]{index=1}

# ---------- 4) Token matching: map ASR words -> your lyrics exactly ----------
def norm_token(s):
    # Remove punctuation, lower, ASCII fold
    s = unidecode(s).lower()
    s = rx.sub(r"[^\p{Letter}\p{Number}']+", "", s)
    return s

def align_words_to_lyrics(lyrics_blocks, asr_words):
    # Prepare a flat list of normalized ASR tokens
    asr = [{"i":i, "w":w["w"], "n":norm_token(w["w"]), "t":w["start"], "t_end":w["end"]} for i,w in enumerate(asr_words)]
    j = 0
    N = len(asr)
    out_blocks = []
    cur_time = 0.0

    for b in lyrics_blocks:
        out_b = {"type": b["type"], "start": None, "end": None, "lines": []}
        for line_group in b["lines"]:
            # line_group is list of word-arrays that were separated by blank lines during parsing
            for words_in_line in line_group:
                tokens = [{"w":w, "n":norm_token(w)} for w in words_in_line]
                line_out = {"start": None, "end": None, "words": []}

                # greedy forward match on normalized tokens
                for tok in tokens:
                    # advance j until match or give up
                    while j < N and asr[j]["n"] == "":
                        j += 1
                    k = j
                    while k < N and asr[k]["n"] != tok["n"]:
                        k += 1
                    if k < N and asr[k]["n"] == tok["n"]:
                        # matched
                        line_out["words"].append({"w": tok["w"], "t": asr[k]["t"]})
                        if line_out["start"] is None:
                            line_out["start"] = asr[k]["t"]
                        line_out["end"] = asr[k]["t_end"]
                        j = k + 1
                    else:
                        # no match: fall back to approximate timing (last known + small delta)
                        fallback_t = (line_out["end"] or cur_time) + 0.20
                        line_out["words"].append({"w": tok["w"], "t": fallback_t})
                        line_out["start"] = line_out["start"] or fallback_t
                        line_out["end"] = fallback_t + 0.15

                cur_time = max(cur_time, line_out["end"] or cur_time)
                out_b["lines"].append(line_out)
                out_b["start"] = out_b["start"] if out_b["start"] is not None else line_out["start"]
                out_b["end"] = line_out["end"]
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
    args = ap.parse_args()

    if not args.aai_key:
        raise SystemExit("Set --aai_key or ASSEMBLYAI_API_KEY")

    # 1) Preprocess audio for robust alignment
    out_wav = os.path.join(os.path.dirname(args.out) or ".", "prep.wav")
    wav_path, sr = prep_audio(args.audio, out_wav)

    # 2) Parse lyrics into blocks/lines
    lyrics_blocks = parse_lyrics(args.lyrics)

    # 3) Upload+transcribe -> per-word times
    words, dur = transcribe_words(wav_path, args.aai_key)

    # 4) Map ASR tokens back to your exact lyrics text
    blocks = align_words_to_lyrics(lyrics_blocks, words)

    # 5) Save timings.json
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(build_output(blocks, sr, "assemblyai", dur), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
