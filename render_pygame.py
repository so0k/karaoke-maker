#!/usr/bin/env python3

import os
import math
import json
import shutil
import subprocess
from typing import List, Tuple, Dict, Any

import numpy as np
import pygame
import librosa
import imageio_ffmpeg


# ---------- Config ----------
W, H = 1920, 1080
FPS = 30
OUT_DIR = "frames_pg"
OUTPUT_MP4 = "karaoke_pygame.mp4"

AUDIO_PATH = "./audio/from_boilerplate_to_flow.mp3"
TIMINGS_JSON = "./audio/from_boilerplate_to_flow_timings.json"

# Colors
BASE_COLOR = (255, 255, 255)
HI_COLOR = (255, 209, 102)  # #FFD166
STROKE_COLOR = (0, 0, 0)
TITLE_COLOR = (160, 231, 229)  # #A0E7E5
BG_COLOR = (12, 16, 24)

FONT_NAME = "DejaVu Sans"
FONT_SIZE = 64
TITLE_FONT_SIZE = 42

MARGIN_BOTTOM = 240
LINE_SPACING = 86

SPRITESHEET = "nes-sprites.png"


# ---------- Helpers ----------
def load_timings(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def flatten_lines(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for b in data.get("blocks", []):
        bl_type = b.get("type", "")
        for ln in b.get("lines", []):
            # handle older nested shape: [ [ {start,end,words}, ... ], ... ]
            if isinstance(ln, list):
                for ln2 in ln:
                    if isinstance(ln2, dict) and "words" in ln2:
                        line = {
                            "type": bl_type,
                            "start": ln2.get("start", 0.0),
                            "end": ln2.get("end", 0.0),
                            "words": ln2.get("words", []),
                        }
                        lines.append(line)
            elif isinstance(ln, dict):
                line = {
                    "type": bl_type,
                    "start": ln.get("start", 0.0),
                    "end": ln.get("end", 0.0),
                    "words": ln.get("words", []),
                }
                lines.append(line)
    return lines


def audio_duration_seconds(path: str) -> float:
    # librosa.get_duration supports filename in newer versions
    try:
        return float(librosa.get_duration(path=path))
    except TypeError:
        y, sr = librosa.load(path, sr=None, mono=True)
        return float(len(y) / sr)


def compute_beats(path: str) -> List[float]:
    y, sr = librosa.load(path, sr=None, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    times = librosa.frames_to_time(beats, sr=sr)
    return times.tolist()


def beat_pulse(t: float, beat_times: List[float], width: float = 0.08) -> float:
    if not beat_times:
        return 0.0
    d = np.abs(np.array(beat_times) - t)
    m = float(d.min())
    return float(math.exp(-(m * m) / (2 * width * width)))


# ---------- Pygame drawing ----------
def load_sprites() -> Dict[str, pygame.Surface]:
    sprites: Dict[str, pygame.Surface] = {}
    try:
        sheet = pygame.image.load(SPRITESHEET).convert_alpha()
        sw, sh = sheet.get_width(), sheet.get_height()
        # Sample a few tiles from the sheet
        tiles: List[Tuple[int, int, int, int]] = [
            (0, 0, 64, 64),
            (64, 0, 64, 64),
            (128, 0, 64, 64),
            (0, 64, 64, 64),
            (64, 64, 64, 64),
        ]
        def crop(rect: Tuple[int, int, int, int]) -> pygame.Surface:
            x, y, w, h = rect
            x = min(max(0, x), sw - 1)
            y = min(max(0, y), sh - 1)
            w = min(w, sw - x)
            h = min(h, sh - y)
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            surf.blit(sheet, (0, 0), (x, y, w, h))
            return surf
        sprites["tile1"] = crop(tiles[0])
        sprites["tile2"] = crop(tiles[1])
        sprites["tile3"] = crop(tiles[2])
        sprites["globe"] = crop((128, 64, 64, 64))
    except Exception:
        # Fallback: simple colored tiles
        def solid(color: Tuple[int, int, int], w_: int = 128, h_: int = 64) -> pygame.Surface:
            s = pygame.Surface((w_, h_), pygame.SRCALPHA)
            s.fill(color)
            return s
        sprites["tile1"] = solid((40, 60, 110))
        sprites["tile2"] = solid((60, 90, 160))
        sprites["tile3"] = solid((80, 120, 210))
        g = pygame.Surface((64, 64), pygame.SRCALPHA)
        pygame.draw.circle(g, (220, 240, 255), (32, 32), 28)
        pygame.draw.circle(g, (160, 180, 210), (32, 32), 28, width=3)
        sprites["globe"] = g
    return sprites


def tile_row(surface: pygame.Surface, sprite: pygame.Surface, y: int, speed_px_s: float, t: float):
    w = sprite.get_width()
    if w <= 0:
        return
    xoff = int((t * speed_px_s) % w)
    x = -xoff
    while x < W:
        surface.blit(sprite, (x, y))
        x += w


def draw_text_with_outline(surface: pygame.Surface, font: pygame.font.Font, text: str, color, outline, x_center: int, y: int):
    # Render outline by multi-blit
    base = font.render(text, True, color)
    outline_surf = font.render(text, True, outline)
    xc = x_center - base.get_width() // 2
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        surface.blit(outline_surf, (xc + dx, y + dy))
    surface.blit(base, (xc, y))


def draw_text_with_outline_left(surface: pygame.Surface, font: pygame.font.Font, text: str, color, outline, x_left: int, y: int):
    # Same as draw_text_with_outline, but anchors on left x instead of centering
    base = font.render(text, True, color)
    outline_surf = font.render(text, True, outline)
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        surface.blit(outline_surf, (x_left + dx, y + dy))
    surface.blit(base, (x_left, y))


def draw_line_highlight(surface: pygame.Surface, font: pygame.font.Font, full_text: str, pre_text: str, y: int):
    # Measure full text to compute its left edge when centered
    full_surface = font.render(full_text, True, BASE_COLOR)
    left_x = (W // 2) - (full_surface.get_width() // 2)

    # Draw base full text centered
    draw_text_with_outline(surface, font, full_text, BASE_COLOR, STROKE_COLOR, W // 2, y)

    # Draw highlighted prefix overlay starting at the same left_x
    if pre_text:
        draw_text_with_outline_left(surface, font, pre_text, HI_COLOR, STROKE_COLOR, left_x, y)


def words_prefix_upto(words: List[Dict[str, Any]], t: float) -> str:
    arr = []
    for w in words:
        if float(w.get("t", 0.0)) <= t:
            arr.append(w.get("w", ""))
    return (" ".join(arr)).strip()


def draw_scene(surface: pygame.Surface, sprites: Dict[str, pygame.Surface], t: float, beat_times: List[float]):
    surface.fill(BG_COLOR)
    tile_row(surface, sprites["tile1"], 240, speed_px_s=20, t=t)
    tile_row(surface, sprites["tile2"], 420, speed_px_s=45, t=t)
    tile_row(surface, sprites["tile3"], 600, speed_px_s=70, t=t)
    # Bobbing orbs on beats
    p = 1.0 + 0.15 * beat_pulse(t, beat_times)
    for (x0, y0, phase) in [(320, 760, 0.0), (960, 720, 0.18), (1600, 780, 0.36)]:
        bob = 6 * math.sin(2 * math.pi * (0.25 * t + phase))
        g = pygame.transform.rotozoom(sprites["globe"], 0, p)
        surface.blit(g, (x0 - g.get_width() // 2, y0 + int(bob) - g.get_height() // 2))


# ---------- ffmpeg helpers ----------
def ffmpeg_has_encoder(name: str) -> bool:
    try:
        # Query encoder help; returns 0 if the encoder exists
        r = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-h", f"encoder={name}"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return r.returncode == 0
    except Exception:
        return False


def choose_ffmpeg_codecs() -> Tuple[str, str]:
    """Return (video_encoder, audio_encoder) with fallbacks."""
    # Prefer software x264, then macOS hardware, then broadly-available mpeg4
    v_candidates = [
        "libx264",
        "h264_videotoolbox",
        "libx265",
        "mpeg4",
    ]
    a_candidates = [
        "aac",
        "libfdk_aac",  # uncommon, but high quality
        "libmp3lame",
    ]

    v = next((enc for enc in v_candidates if ffmpeg_has_encoder(enc)), None)
    a = next((enc for enc in a_candidates if ffmpeg_has_encoder(enc)), None)

    # Best-effort defaults if detection failed; ffmpeg will error if unsupported
    if v is None:
        v = "mpeg4"
    if a is None:
        a = "aac"
    return v, a


def get_ffmpeg_bin() -> str:
    """Prefer MoviePy/imageio's bundled ffmpeg to avoid PATH differences."""
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def render_frames():
    # Prep IO
    os.makedirs(OUT_DIR, exist_ok=True)

    # Timings and beats
    data = load_timings(TIMINGS_JSON)
    lines = flatten_lines(data)
    dur_audio = audio_duration_seconds(AUDIO_PATH)
    # Extend duration a bit beyond last word if shorter than audio
    last_word = 0.0
    for L in lines:
        for w in L.get("words", []):
            tw = float(w.get("t", 0.0))
            if tw > last_word:
                last_word = tw
    duration = max(dur_audio, last_word + 0.5)

    # Beats
    try:
        beat_times = compute_beats(AUDIO_PATH)
    except Exception:
        beat_times = []

    # Pygame init
    pygame.init()
    pygame.font.init()
    screen = pygame.Surface((W, H))
    sprites = load_sprites()
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
    title_font = pygame.font.SysFont(FONT_NAME, TITLE_FONT_SIZE)

    # Precompute positions
    y_main = H - MARGIN_BOTTOM
    y_prev = y_main - LINE_SPACING

    # Frame loop
    n_frames = int(duration * FPS)
    print(f"Rendering {n_frames} frames @ {FPS} fps (~{duration:.2f}s)")

    for i in range(n_frames):
        t = i / FPS

        # Background
        draw_scene(screen, sprites, t, beat_times)

        # Current block type (for title)
        current_block = None
        for L in lines:
            if float(L.get("start", 0.0)) <= t <= float(L.get("end", 0.0)):
                current_block = L.get("type", None)
                break
        if current_block:
            draw_text_with_outline(screen, title_font, str(current_block).title(), TITLE_COLOR, STROKE_COLOR, 140, 60)

        # Visible lines
        visible = [
            L for L in lines if (float(L.get("start", 0.0)) - 0.25) <= t <= (float(L.get("end", 0.0)) + 0.25)
        ]
        visible.sort(key=lambda L: float(L.get("start", 0.0)))
        if len(visible) > 2:
            visible = visible[-2:]

        if visible:
            if len(visible) == 1:
                v = visible[0]
                full = " ".join([w.get("w", "") for w in v.get("words", [])])
                pre = words_prefix_upto(v.get("words", []), t)
                draw_line_highlight(screen, font, full, pre, y_main)
            else:
                v0, v1 = visible[-2], visible[-1]
                full0 = " ".join([w.get("w", "") for w in v0.get("words", [])])
                pre0 = words_prefix_upto(v0.get("words", []), t)
                draw_line_highlight(screen, font, full0, pre0, y_prev)

                full1 = " ".join([w.get("w", "") for w in v1.get("words", [])])
                pre1 = words_prefix_upto(v1.get("words", []), t)
                draw_line_highlight(screen, font, full1, pre1, y_main)

        # Save frame
        out_path = os.path.join(OUT_DIR, f"{i:06d}.png")
        pygame.image.save(screen, out_path)

        if i % 150 == 0:
            print(f"Frame {i}/{n_frames} ({100.0*i/n_frames:.1f}%)")

    pygame.quit()

    # Encode with ffmpeg if available
    ffmpeg_bin = get_ffmpeg_bin()
    if ffmpeg_bin:
        venc, aenc = choose_ffmpeg_codecs()

        # Build command with sensible flags per encoder
        input_pattern = os.path.join(OUT_DIR, "%06d.png")
        cmd = [
            ffmpeg_bin, "-y",
            "-hide_banner", "-loglevel", "warning",
            "-framerate", str(FPS),  # proper input rate for image2
            "-i", input_pattern,
            "-i", AUDIO_PATH,
            "-c:v", venc,
        ]

        # Video encoder specific tuning
        if venc == "libx264":
            cmd += ["-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1", "-crf", "18", "-preset", "medium"]
        elif venc == "h264_videotoolbox":
            # macOS hardware encoder; bitrate-based
            cmd += ["-b:v", "6M", "-maxrate", "8M", "-pix_fmt", "yuv420p"]
        elif venc == "libx265":
            cmd += ["-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-crf", "22", "-preset", "medium"]
        elif venc == "mpeg4":
            cmd += ["-q:v", "5", "-pix_fmt", "yuv420p"]

        # Audio encoder and common flags
        cmd += [
            "-c:a", aenc,
        ]
        # Bitrate if using CBR-style encoders
        if aenc in ("aac", "libfdk_aac", "libmp3lame"):
            cmd += ["-b:a", "192k"]

        cmd += [
            "-shortest",
            "-movflags", "+faststart",
            "-r", str(FPS),  # output frame rate
            OUTPUT_MP4,
        ]

        print(f"Encoding MP4 with ffmpeg using {venc}/{aenc} …")
        try:
            subprocess.run(cmd, check=True)
            print(f"Wrote {OUTPUT_MP4}")
        except Exception as e:
            print("ffmpeg encode failed:", e)
            print("You can run this manually:")
            print(" ", " ".join(cmd))
            print("If libx264 is unavailable, try h264_videotoolbox or mpeg4, e.g.:")
            print(
                f"  ffmpeg -y -framerate {FPS} -i {OUT_DIR}/%06d.png -i {AUDIO_PATH} -c:v h264_videotoolbox -b:v 6M -pix_fmt yuv420p -c:a aac -b:a 192k -shortest -movflags +faststart {OUTPUT_MP4}"
            )
            print(
                f"  ffmpeg -y -framerate {FPS} -i {OUT_DIR}/%06d.png -i {AUDIO_PATH} -c:v mpeg4 -q:v 5 -pix_fmt yuv420p -c:a aac -b:a 192k -shortest -movflags +faststart {OUTPUT_MP4}"
            )
    else:
        print("ffmpeg not found. To encode, run:")
        print(f"ffmpeg -y -framerate {FPS} -i {OUT_DIR}/%06d.png -i {AUDIO_PATH} -c:v h264_videotoolbox -b:v 6M -pix_fmt yuv420p -c:a aac -b:a 192k -shortest -movflags +faststart {OUTPUT_MP4}")


if __name__ == "__main__":
    render_frames()
