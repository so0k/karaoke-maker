# pip install moviepy==1.0.3 numpy pillow
# Also install ImageMagick so TextClip can render text:
#   macOS: brew install imagemagick
#   Ubuntu/Debian: sudo apt-get install imagemagick
#   Windows: install, then ensure MoviePy sees it (config)  # docs below

from moviepy import (
    TextClip, CompositeVideoClip, VideoClip, AudioFileClip, ColorClip
)
# from moviepy.config import change_settings
# change_settings({"IMAGEMAGICK_BINARY": "/opt/homebrew/bin/magick"})
import numpy as np, json, math, os

# ---------- Config ----------
W, H   = 1920, 1080
FPS    = 30
BASE   = "#FFFFFF"   # unsung text
HI     = "#FFD166"   # highlighted text
STROKE = 4
FONT   = "DejaVu-Sans"   # swap for a TTF installed on your system
TITLE_FONT = "DejaVu-Sans"
TITLE_COLOR = "#A0E7E5"

MARGIN_BOTTOM = 240  # vertical offset for lyrics
LINE_SPACING  = 86   # distance between the two lines

# ---------- Load assets ----------
audio = AudioFileClip("./audio/from_boilerplate_to_flow.mp3")  # original mix
data  = json.load(open("./audio/from_boilerplate_to_flow_timings.json", "r", encoding="utf-8"))

# Optional: you can compute total duration from audio
duration = audio.duration

# ---------- 8-bit background ----------
def make_pixel_bg(t, w=W, h=H):
    """
    Two parallax layers of chunky tiles that slowly scroll.
    Pure NumPy -> fast and no external textures.
    """
    # base palette channels that drift over time
    t1 = t * 12.0
    t2 = t * 7.0

    # choose 'pixel size' (scale up to look 8-bit)
    px = 16  # bigger -> chunkier
    gw, gh = w // px, h // px

    # generate a moving checker for layer A
    xa = (np.arange(gw)[None, :] + int(t1) // 3) % 2
    ya = (np.arange(gh)[:, None] + int(t1) // 4) % 2
    layer_a = (xa ^ ya).astype(np.float32)

    # layer B diagonal stripes
    xb = (np.arange(gw)[None, :] + int(t2) // 2)
    yb = (np.arange(gh)[:, None] + int(t2) // 3)
    layer_b = (((xb + yb) // 2) % 2).astype(np.float32)

    # mix to RGB (soft palette shifts)
    r = 40  + 40*layer_a + 35*layer_b + 25*np.sin(0.6*t)
    g = 30  + 45*layer_a + 30*layer_b + 25*np.sin(0.7*t + 1.0)
    b = 60  + 30*layer_a + 50*layer_b + 25*np.sin(0.8*t + 2.0)

    img = np.clip(np.dstack([r,g,b]), 0, 255).astype(np.uint8)

    # nearest-neighbor upscale to screen size
    frame = np.repeat(np.repeat(img, px, axis=0), px, axis=1)
    return frame[:h, :w, :]

bg = VideoClip(make_frame=make_pixel_bg, duration=duration).set_fps(FPS)

# ---------- Text helpers ----------
def make_line_clip(full_text, pre_highlight_text):
    """
    Render one lyric line by stacking:
      - base full line (white)
      - an overlay clipping the highlighted prefix width (accent)
    This avoids reflow while we progressively color text.
    """
    base = TextClip(
        full_text, font=FONT, fontsize=70, color=BASE,
        stroke_color="black", stroke_width=STROKE, method="label"
    )
    if not pre_highlight_text:
        return base

    # Render the prefix in highlight color, then paste it over as a width-cropped overlay
    hi   = TextClip(
        pre_highlight_text, font=FONT, fontsize=70, color=HI,
        stroke_color="black", stroke_width=STROKE, method="label"
    )
    # Put both at same position in a same-sized canvas, crop overlay to its own width
    overlay = hi.on_color(size=base.size, color=None, col_opacity=0)
    overlay = overlay.crop(x1=0, y1=0, x2=hi.w, y2=hi.h)
    return CompositeVideoClip([base, overlay], size=base.size)

def words_upto_now(words, t_now):
    """
    Build two strings:
      pre_highlight: words whose timestamp <= t_now
      post: remaining (unused here, since we draw base full line)
    """
    pre = []
    for w in words:
        if w["t"] <= t_now:
            pre.append(w["w"])
    return (" ".join(pre)).strip()

# ---------- Lyric layer (max 2 lines) ----------
def lyric_frame_factory(data):
    # Flatten blocks into a list of (block_type, line) with timings
    lines = []
    for b in data["blocks"]:
        for ln in b["lines"]:
            lines.append({
                "type": b["type"],
                "start": ln["start"],
                "end": ln["end"],
                "words": ln["words"]
            })

    # Precompute y positions for up to 2 lines
    y_main = H - MARGIN_BOTTOM
    y_prev = y_main - LINE_SPACING

    def make_frame(t):
        # find visible lines around t
        visible = [L for L in lines if (L["start"] - 0.25) <= t <= (L["end"] + 0.25)]
        visible = sorted(visible, key=lambda L: L["start"])

        # choose at most two: the current and the next/prev relevant
        if len(visible) > 2:
            visible = visible[-2:]

        clips = []
        # Block title - show the current block type for 1.0s when it changes
        current_block = None
        for L in lines:
            if L["start"] <= t <= L["end"]:
                current_block = L["type"]
                break

        if current_block:
            title = TextClip(
                current_block.title(), font=TITLE_FONT, fontsize=48,
                color=TITLE_COLOR, stroke_color="black", stroke_width=2, method="label"
            ).with_position((60, 60)).with_duration(1/ FPS)
            # gentle alpha depending on how close to a block boundary
            # (for simplicity we just show it whenever within a line of that block)
            clips.append(title)

        # Render visible lines
        if visible:
            # Bottom row is the "most current" line
            # If 2 lines: earlier line sits above
            if len(visible) == 1:
                v = visible[0]
                full = " ".join([w["w"] for w in v["words"]])
                pre  = words_upto_now(v["words"], t)
                row  = make_line_clip(full, pre).with_position(("center", y_main))
                clips.append(row)
            else:
                v0, v1 = visible[-2], visible[-1]
                full0 = " ".join([w["w"] for w in v0["words"]])
                pre0  = words_upto_now(v0["words"], t)
                row0  = make_line_clip(full0, pre0).with_position(("center", y_prev))

                full1 = " ".join([w["w"] for w in v1["words"]])
                pre1  = words_upto_now(v1["words"], t)
                row1  = make_line_clip(full1, pre1).with_position(("center", y_main))

                clips += [row0, row1]

        if not clips:
            # nothing to render - transparent frame
            return np.zeros((H, W, 3), dtype=np.uint8)

        comp = CompositeVideoClip(clips, size=(W, H)).with_duration(1 / FPS)
        return comp.get_frame(0)

    return VideoClip(make_frame=make_frame, duration=duration)

lyrics_layer = lyric_frame_factory(data)

# ---------- Compose and render ----------
final = CompositeVideoClip(
    [bg, lyrics_layer.with_position((0,0))],
    size=(W, H)
).with_audio(audio)

# H.264 + AAC + web-friendly flags
final.write_videofile(
    "karaoke_8bit.mp4",
    fps=FPS,
    codec="libx264",
    audio_codec="aac",
    bitrate="6000k",
    ffmpeg_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"]
)
