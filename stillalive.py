"""
StillAlive-inspired terminal music video in pure MoviePy v2.

Left pane: typewriter-style lyrics with a blinking cursor, grouped by paragraphs
           from audio/from_boilerplate_to_flow_lyrics.txt and timed using
           audio/from_boilerplate_to_flow_timings.json.

Right pane: faux `terraform plan` scroll, with <error> (red) and <bliss> (cyan)
            sections, styled like a console.

Outputs: stillalive.mp4

Requires: moviepy>=2.2, pillow, numpy (see requirements.txt)
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from moviepy import VideoClip, ImageClip, TextClip, CompositeVideoClip, AudioFileClip, ColorClip


# ---------------------- Config ----------------------
W, H = 1920, 1080
FPS = 30

# Pane layout
LEFT_W = 1280
RIGHT_W = W - LEFT_W  # 640
PADDING = 24
# Left-pane specific horizontal padding (to maximize width)
LEFT_PAD = 12

# Colors (terminal vibe)
BG = (6, 8, 10)
FG = "#a8ff60"        # terminal green
DIM = "#6aa84f"       # dimmer green
ERR = "#ff6b6b"       # error red
BLISS = "#b388ff"     # purple for <bliss>
FRAME = (32, 64, 48)  # frame lines

# Fonts
MONO = os.environ.get("STILLALIVE_MONO", "Menlo")  # set env var for a specific TTF path if needed
FONT_SIZE = 32
LINE_GAP = 8

# Typewriter rate
CPS = 24  # chars per second

# Plan pane
PLAN_FONT_SIZE = 18
PLAN_LINES_PER_SEC = 14.0   # line reveal rate (no scrolling)
PLAN_CHUNK_PAUSE = 3.0      # hold last lines visible (seconds)
PLAN_CLEAR_PAUSE = 0.7      # blank between chunks (seconds)


LYRICS_PATH = os.path.join("audio", "from_boilerplate_to_flow_lyrics.txt")
TIMINGS_PATH = os.path.join("audio", "from_boilerplate_to_flow_timings.json")
AUDIO_PATH = os.path.join("audio", "from_boilerplate_to_flow.mp3")


# ---------------------- Helpers ----------------------
def safe_text(text: str, font: str | None, font_size: int, color: str,
              method: str = "caption", size: Tuple[int, int] | None = None,
              stroke_color: str | None = "black", stroke_width: int = 1,
              text_align: str = "left", margin: Tuple[int, int] | None = None,
              horizontal_align: str = "left", vertical_align: str = "top") -> TextClip:
    """Create a TextClip with a graceful font fallback if the requested font is missing."""
    kw = dict(
        text=text,
        font=font,
        font_size=font_size,
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        method=method,
        text_align=text_align,
        horizontal_align=horizontal_align,
        vertical_align=vertical_align,
    )
    if size is not None:
        kw["size"] = size
    if margin is not None:
        kw["margin"] = margin
    try:
        return TextClip(**kw)
    except Exception:
        kw["font"] = None
        return TextClip(**kw)


def normalize_line(s: str) -> str:
    """Normalize a lyric/timing line for matching across punctuation/case variants."""
    s = s.strip().lower()
    # unify quotes/dashes
    s = s.replace("’", "'").replace("—", "-").replace("–", "-")
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass
class TimingLine:
    start: float
    end: float
    text: str


@dataclass
class Paragraph:
    lines: List[TimingLine]
    start: float
    end: float
    # adjusted timeline
    start_adj: float = 0.0
    visible_end: float = 0.0


def load_timing_lines(path: str) -> List[TimingLine]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines: List[TimingLine] = []
    for b in data.get("blocks", []):
        for ln in b.get("lines", []):
            words = ln.get("words", [])
            text = " ".join(w.get("w", "") for w in words).strip()
            if not text:
                continue
            lines.append(TimingLine(start=ln.get("start", 0.0), end=ln.get("end", 0.0), text=text))
    return lines


def load_lyric_paragraphs(path: str) -> List[List[str]]:
    """Return paragraphs as lists of lines (excluding [Section] markers)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    paras: List[List[str]] = []
    cur: List[str] = []
    for line in raw:
        if not line.strip():
            if cur:
                paras.append(cur)
                cur = []
            continue
        if line.strip().startswith("[") and line.strip().endswith("]"):
            # skip section headers
            continue
        cur.append(line.rstrip())
    if cur:
        paras.append(cur)
    return paras


def map_paragraph_times(paragraphs: List[List[str]], timing_lines: List[TimingLine]) -> List[Paragraph]:
    """
    Align lyric paragraphs to timing lines by sequential text matching.
    For each paragraph, set start = first matched line start, end = last matched line end.
    """
    mapped: List[Paragraph] = []
    t_idx = 0
    for para in paragraphs:
        para_norm_lines = [normalize_line(s) for s in para]
        p_lines: List[TimingLine] = []
        para_start = None
        para_end = None
        for ln_orig, ln in zip(para, para_norm_lines):
            matched = False
            while t_idx < len(timing_lines):
                cand = timing_lines[t_idx]
                cand_norm = normalize_line(cand.text)
                t_idx += 1
                if ln == cand_norm or ln in cand_norm or cand_norm in ln:
                    tl = TimingLine(start=cand.start, end=cand.end, text=ln_orig)
                    p_lines.append(tl)
                    para_start = cand.start if para_start is None else para_start
                    para_end = cand.end
                    matched = True
                    break
            if not matched:
                # synthetic timing if unmatched
                last_end = p_lines[-1].end if p_lines else (mapped[-1].end if mapped else 0.0)
                synth_start = last_end + 0.4
                synth_end = synth_start + max(1.6, len(ln_orig) / max(12.0, CPS))
                tl = TimingLine(start=synth_start, end=synth_end, text=ln_orig)
                p_lines.append(tl)
                para_start = synth_start if para_start is None else para_start
                para_end = synth_end
        if para_start is None:
            last_end = mapped[-1].end if mapped else 0.0
            para_start = last_end + 0.2
            para_end = para_start + 2.0
        mapped.append(Paragraph(lines=p_lines, start=float(para_start), end=float(para_end)))
    return mapped


# ---------------------- Left Pane: Typewriter ----------------------
def render_left_frame_factory(paragraphs: List[Paragraph]):
    pane_w, pane_h = LEFT_W, H
    wrap_w = pane_w - 2 * LEFT_PAD

    # Timing parameters
    LINE_LEAD = 1.0  # start typing 1s before first word
    MIN_CPS = 14.0
    TYPE_SPEEDUP = 1.15  # slight speed increase (~15%)

    # Flatten lines and compute adjusted starts
    all_lines: List[dict] = []
    for p in paragraphs:
        delta = (p.start_adj if p.start_adj else p.start) - p.start
        for ln in p.lines:
            start_real = ln.start + delta
            end_real = ln.end + delta
            type_start = start_real - LINE_LEAD
            dur = max(0.6, end_real - type_start - 0.05)
            cps_eff = max(MIN_CPS, len(ln.text) / dur) * TYPE_SPEEDUP
            all_lines.append({
                "text": ln.text,
                "type_start": type_start,
                "end_real": end_real,
                "cps": cps_eff,
            })

    # Sort by start time to ensure correct order
    all_lines.sort(key=lambda d: d["type_start"])

    # Cache snapshots (rgb, mask) for lines once they move up (i.e., after the next line starts)
    snapshots: dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def alpha_blit(dest, fr, mask, x, y):
        h, w = fr.shape[:2]
        Hd, Wd = dest.shape[:2]
        if x >= Wd or y >= Hd:
            return
        # Crop if y < 0 or x < 0
        y0 = y
        x0 = x
        fy0 = 0
        fx0 = 0
        if y0 < 0:
            fy0 = -y0
            h = h - fy0
            y0 = 0
        if x0 < 0:
            fx0 = -x0
            w = w - fx0
            x0 = 0
        if h <= 0 or w <= 0:
            return
        # Clamp to destination bounds
        h = min(h, Hd - y0)
        w = min(w, Wd - x0)
        if h <= 0 or w <= 0:
            return
        sub = dest[y0 : y0 + h, x0 : x0 + w, :]
        a = mask[fy0 : fy0 + h, fx0 : fx0 + w]
        if a.size == 0 or a.max() <= 0:
            return
        a = a.astype(np.float32)[..., None]
        sub[:] = (a * fr[fy0 : fy0 + h, fx0 : fx0 + w, :].astype(np.float32) + (1.0 - a) * sub.astype(np.float32)).clip(0, 255).astype(np.uint8)

    def frame_at(t: float) -> np.ndarray:
        im = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)
        im[:, :, :] = BG

        if not all_lines:
            return im

        # Determine current line index by start times
        i_cur = 0
        for i, d in enumerate(all_lines):
            if d["type_start"] <= t:
                i_cur = i
            else:
                break

        # Current line: render typed portion with wrapping to compute true height
        d = all_lines[i_cur]
        elapsed_cur = max(0.0, t - d["type_start"])
        k = max(0, min(len(d["text"]), int(elapsed_cur * d["cps"])) )
        # Snap to full length slightly before annotated end to ensure cursor reaches the end
        if t >= d["end_real"] - 0.05:
            k = len(d["text"])
        vis = d["text"][:k]
        tc = safe_text(vis or " ", MONO, FONT_SIZE, FG,
                       method="caption", size=(wrap_w, None), text_align="left",
                       margin=(0, int(FONT_SIZE * 0.5)))
        fr_text = tc.with_duration(1 / FPS).get_frame(0)
        mk_text = tc.to_mask().with_duration(1 / FPS).get_frame(0)
        # Vertical center based on current rendered height
        y_center_top = (pane_h - fr_text.shape[0]) // 2

        # Build snapshots for previous lines (full wrapped content), and stack above
        prev_idx = i_cur - 1
        y_above = y_center_top - LINE_GAP
        while prev_idx >= 0:
            if prev_idx not in snapshots:
                dprev = all_lines[prev_idx]
                tc_prev = safe_text(dprev["text"] or " ", MONO, FONT_SIZE, FG,
                                    method="caption", size=(wrap_w, None), text_align="left",
                                    margin=(0, int(FONT_SIZE * 0.5)))
                fr_prev = tc_prev.with_duration(1 / FPS).get_frame(0)
                mk_prev = tc_prev.to_mask().with_duration(1 / FPS).get_frame(0)
                snapshots[prev_idx] = (fr_prev, mk_prev)
            fr_prev, mk_prev = snapshots[prev_idx]
            y_above -= fr_prev.shape[0]
            alpha_blit(im, fr_prev, mk_prev, LEFT_PAD, y_above)
            y_above -= LINE_GAP
            prev_idx -= 1

        # Render text to a canvas as wide as wrap_w, then draw cursor slightly ahead at end-of-text
        canvas_w = wrap_w
        canvas_h = fr_text.shape[0]
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :, :] = BG
        canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        # blit text at left
        alpha_blit(canvas, fr_text, mk_text, 0, 0)

        # cursor position slightly ahead of the last glyph of the wrapped text
        show_cur = (int(t * 2) % 2 == 0)
        if show_cur:
            ch = max(2, int(FONT_SIZE * 0.12))
            cw = max(8, int(FONT_SIZE * 0.6))
            gap = max(4, int(FONT_SIZE * 0.2))
            ys, xs = np.where(mk_text > 0)
            if ys.size > 0:
                last_row = int(ys.max())
                row_mask = mk_text[last_row, :]
                cols = np.where(row_mask > 0)[0]
                last_col = int(cols.max()) if cols.size > 0 else fr_text.shape[1] - 1
                x0 = min(last_col + gap, canvas_w - cw - 1)
                y0 = min(last_row + 1, canvas_h - ch - 1)
            else:
                x0 = 0
                y0 = canvas_h - ch - 1
            canvas[y0:y0 + ch, x0:x0 + cw, :] = np.array([168, 255, 96], dtype=np.uint8)
            canvas_mask[y0:y0 + ch, x0:x0 + cw] = 1.0

        # build mask that includes text and cursor; crop to canvas bounds
        h = min(mk_text.shape[0], canvas_mask.shape[0])
        w = min(mk_text.shape[1], canvas_mask.shape[1])
        if h > 0 and w > 0:
            canvas_mask[:h, :w] = np.maximum(canvas_mask[:h, :w], mk_text[:h, :w])

        # center on screen vertically, left aligned horizontally
        alpha_blit(im, canvas, canvas_mask, LEFT_PAD, y_center_top)

        return im

    return frame_at


# ---------------------- Right Pane: Terraform Scroll ----------------------
PLAN_BLOCKS = [
    # Block 1 — Initializing Chaos (error period)
    (
        """
Acquiring state lock. This may take a few moments...
Initializing the backend...
Initializing provider plugins...
- Finding hashicorp/aws versions matching ~> 5.0...
- Installing hashicorp/aws v5.53.0...
- Installed hashicorp/aws v5.53.0 (signed by HashiCorp)

Refreshing Terraform state in-memory prior to plan...
module.network.aws_vpc.core: Refreshing state...
module.network.aws_subnet.public[0]: Refreshing state...
module.network.aws_subnet.public[1]: Refreshing state...
module.network.aws_route_table.public: Refreshing state...
module.edge.aws_cloudfront_distribution.cdn: Refreshing state...
module.storage.aws_s3_bucket.logs: Refreshing state...
module.compute.aws_lambda_function.ingestor: Refreshing state...
module.compute.aws_iam_role.lambda_exec: Refreshing state...
module.observability.aws_cloudwatch_log_group.app: Refreshing state...
module.db.aws_rds_cluster.main: Refreshing state...

<error>Error: Invalid attribute name in locals
  on locals.tf line 7:
  locals { tagz = { "owner" = "ops" } }
  Expected "tags", got "tagz". Did a stray bracket ruin our night again?</error>

<error>Plan aborted due to 1 error. Fix configuration, then run 'terraform plan' again.</error>
"""
    ).strip(),
    # Block 2 — Grinding Through Boilerplate (more refresh spam)
    (
        """
Refreshing Terraform state in-memory prior to plan...
module.core.aws_iam_policy.readonly: Refreshing state...
module.core.aws_iam_user.deployer: Refreshing state...
module.edge.aws_route53_record.api: Refreshing state...
module.edge.aws_acm_certificate.site: Refreshing state...
module.compute.aws_security_group.web[0]: Refreshing state...
module.compute.aws_security_group.web[1]: Refreshing state...
module.compute.aws_launch_template.web: Refreshing state...
module.compute.aws_autoscaling_group.web: Refreshing state...
module.storage.aws_s3_bucket.assets: Refreshing state...
module.storage.aws_s3_bucket_versioning.assets: Refreshing state...
module.ci.aws_iam_role.github_actions: Refreshing state...
module.ci.aws_iam_role_policy.attachments["s3"]: Refreshing state...
module.network.aws_nat_gateway.main: Refreshing state...
module.network.aws_eip.nat: Refreshing state...
module.network.aws_route.public_internet: Refreshing state...

<error>Error: Count cannot be computed
  on modules/compute/main.tf line 42, in resource "aws_instance" "runner":
  count = var.enable_runners ? var.runner_count : 0
  runner_count depends on unknown value (stop copy-pasting that block).</error>

<error>Hint: try 'terraform validate' and stop wrestling HCL in endless lines.</error>
"""
    ).strip(),
    # Block 3 — Turning the Corner (first clean plan)
    (
        """
Refreshing Terraform state in-memory prior to plan...
module.edge.aws_cloudfront_origin_access_identity.default: Refreshing state...
module.edge.aws_cloudfront_cache_policy.default: Refreshing state...
module.network.aws_security_group.lb: Refreshing state...
module.network.aws_lb.application: Refreshing state...
module.network.aws_lb_listener.https: Refreshing state...
module.compute.aws_lambda_function.ingestor: Refreshing state...
module.compute.aws_lambda_permission.cdn_invoke: Refreshing state...
module.storage.aws_s3_bucket_state.terraform: Refreshing state...
module.storage.aws_dynamodb_table.locks: Refreshing state...
module.ops.aws_iam_policy.footguns_denied: Refreshing state...
module.ops.aws_iam_role.guardrails: Refreshing state...

<bliss>Success: Configuration validated. No syntax errors detected.</bliss>
<bliss>Plan: 3 to add, 1 to change, 0 to destroy.</bliss>

  # module.ops.aws_iam_policy.footguns_denied will be created
  + resource "aws_iam_policy" "footguns_denied" { ... }

  # module.edge.aws_cloudfront_distribution.cdn will be updated in-place
  ~ cache_behavior[0].min_ttl = 0 -> 60

  # module.compute.aws_lambda_function.ingestor will be created
  + resource "aws_lambda_function" "ingestor" { runtime = "python3.13" ... }

  # module.storage.aws_s3_bucket.assets will be created
  + resource "aws_s3_bucket" "assets" { bucket = "tcons-assets-xyz" ... }
"""
    ).strip(),
    # Block 4 — Apply With Style
    (
        """
Applying Terraform configuration...
module.ops.aws_iam_policy.footguns_denied: Creating...
module.ops.aws_iam_policy.footguns_denied: Creation complete after 2s [id=arn:aws:iam::123456789012:policy/footguns_denied]
module.edge.aws_cloudfront_distribution.cdn: Modifying... [id=E28CLOUDMAGIC]
module.edge.aws_cloudfront_distribution.cdn: Modifications complete after 14s
module.compute.aws_lambda_function.ingestor: Creating...
module.compute.aws_lambda_function.ingestor: Still creating... [10s elapsed]
module.compute.aws_lambda_function.ingestor: Creation complete after 12s [id=ingestor-terra]
module.storage.aws_s3_bucket.assets: Creating...
module.storage.aws_s3_bucket.assets: Creation complete after 3s [id=tcons-assets-xyz]
Outputs:
  assets_bucket = "s3://tcons-assets-xyz"
  cdn_domain    = "d3k4r40.example.cloudfront.net"

<bliss>Apply complete! Resources: 3 added, 1 changed, 0 destroyed.</bliss>
<bliss>Infrastructure state is clean, clear, and cloud-locked.</bliss>
"""
    ).strip(),
    # Block 5 — Glow-up Montage
    (
        """
Refreshing Terraform state in-memory prior to plan...
module.mesh.aws_vpc_lattice_service.network: Refreshing state...
module.mesh.aws_vpc_lattice_service_auth_policy.strict: Refreshing state...
module.observability.aws_xray_sampling_rule.default: Refreshing state...
module.observability.aws_cloudwatch_dashboard.ops: Refreshing state...
module.edge.aws_route53_record.www: Refreshing state...
module.edge.aws_wafv2_web_acl.main: Refreshing state...
module.ci.aws_ecr_repository.images: Refreshing state...
module.ci.aws_iam_role.deploy: Refreshing state...
module.runtime.aws_eks_cluster.main: Refreshing state...
module.runtime.aws_eks_node_group.app[0]: Refreshing state...
module.runtime.aws_eks_node_group.app[1]: Refreshing state...
module.runtime.aws_eks_addon.vpc-cni: Refreshing state...
module.runtime.aws_eks_addon.coredns: Refreshing state...
module.runtime.aws_eks_addon.kube-proxy: Refreshing state...
module.runtime.kubernetes_namespace.platform: Refreshing state...

<bliss>No changes. Your infrastructure matches the configuration.</bliss>
<bliss>Plan: 0 to add, 0 to change, 0 to destroy.</bliss>
Milestone: Cloud architecture unlocked!
>(_^◡^)_🎉
"""
    ).strip(),
]


def parse_plan_lines(blocks: List[str]) -> List[Tuple[str, str]]:
    """Flatten blocks into lines tagged with color category based on <error>/<bliss> tags."""
    out: List[Tuple[str, str]] = []
    for blk in blocks:
        i = 0
        while i < len(blk):
            m_err = re.search(r"<error>(.*?)</error>", blk[i:], flags=re.S)
            m_bliss = re.search(r"<bliss>(.*?)</bliss>", blk[i:], flags=re.S)
            m = None
            tag = None
            if m_err and (not m_bliss or m_err.start() < m_bliss.start()):
                m = m_err
                tag = "error"
            elif m_bliss:
                m = m_bliss
                tag = "bliss"

            if not m:
                segment = blk[i:]
                lines = segment.splitlines()
                for ln in lines:
                    out.append((ln, "normal"))
                break

            # pre-tag text
            pre = blk[i : i + m.start()]
            for ln in pre.splitlines():
                out.append((ln, "normal"))
            # tagged text
            inner = m.group(1)
            for ln in inner.splitlines():
                out.append((ln, tag))
            i = i + m.end()
        # blank line between blocks
        out.append(("", "normal"))
    return out


def parse_block_lines(block: str) -> List[Tuple[str, str]]:
    """Parse a single block into (line, tag) tuples, honoring <error>/<bliss>."""
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(block):
        m_err = re.search(r"<error>(.*?)</error>", block[i:], flags=re.S)
        m_bliss = re.search(r"<bliss>(.*?)</bliss>", block[i:], flags=re.S)
        m = None
        tag = None
        if m_err and (not m_bliss or m_err.start() < m_bliss.start()):
            m = m_err
            tag = "error"
        elif m_bliss:
            m = m_bliss
            tag = "bliss"

        if not m:
            segment = block[i:]
            lines = segment.splitlines()
            for ln in lines:
                out.append((ln, "normal"))
            break

        pre = block[i : i + m.start()]
        for ln in pre.splitlines():
            out.append((ln, "normal"))
        inner = m.group(1)
        for ln in inner.splitlines():
            out.append((ln, tag))
        i = i + m.end()
    return out


def prerender_plan_surface(lines: List[Tuple[str, str]], width: int) -> np.ndarray:
    surf_w = width
    # Estimate height: 40px per line + padding
    est_h = max(H * 2, 40 * (len(lines) + 20))
    img = np.zeros((est_h, surf_w, 3), dtype=np.uint8)
    img[:, :, :] = BG

    y = PADDING
    max_line_w = max(10, surf_w - 2 * PADDING)
    for text, tag in lines:
        color = FG
        if tag == "error":
            color = ERR
        elif tag == "bliss":
            color = BLISS
        # Dim empty lines slightly via color choice
        col = color if text.strip() else DIM
        # No wrapping: render single line and clip to pane width
        tc = safe_text(text or " ", MONO, PLAN_FONT_SIZE, col, method="label", text_align="left")
        fr = tc.with_duration(1 / FPS).get_frame(0)
        mk = tc.to_mask().with_duration(1 / FPS).get_frame(0)
        if y + fr.shape[0] + 8 >= est_h:
            break
        # Alpha blit onto surface to avoid background shade mismatch
        h, w = fr.shape[:2]
        w = min(w, surf_w - 2 * PADDING)
        sub = img[y : y + h, PADDING : PADDING + w, :]
        a = mk[:h, :w].astype(np.float32)[..., None]
        sub[:] = (a * fr[:h, :w, :].astype(np.float32) + (1.0 - a) * sub.astype(np.float32)).clip(0, 255).astype(np.uint8)
        y += fr.shape[0] + 8
    return img[: max(y + 40, H), :, :]


# ---------------------- Frame Decorations ----------------------
def frame_overlay(size: Tuple[int, int], left_w: int, right_w: int) -> ImageClip:
    """Return an ImageClip with pane frames drawn as simple rectangles, with mask transparency."""
    w, h = size
    arr_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    arr_mask = np.zeros((h, w), dtype=np.uint8)

    def draw_rect(x, y, ww, hh, color=(32, 64, 48), thickness=2):
        r, g, b = color
        # top/bottom
        arr_rgb[y : y + thickness, x : x + ww, :3] = (r, g, b)
        arr_mask[y : y + thickness, x : x + ww] = 255
        arr_rgb[y + hh - thickness : y + hh, x : x + ww, :3] = (r, g, b)
        arr_mask[y + hh - thickness : y + hh, x : x + ww] = 255
        # left/right
        arr_rgb[y : y + hh, x : x + thickness, :3] = (r, g, b)
        arr_mask[y : y + hh, x : x + thickness] = 255
        arr_rgb[y : y + hh, x + ww - thickness : x + ww, :3] = (r, g, b)
        arr_mask[y : y + hh, x + ww - thickness : x + ww] = 255

    # outer border
    draw_rect(0, 0, w, h, FRAME, thickness=3)
    # left pane
    draw_rect(0, 0, left_w, h, FRAME, thickness=2)
    # right pane
    draw_rect(left_w, 0, right_w, h, FRAME, thickness=2)

    overlay = ImageClip(arr_rgb, transparent=True).with_duration(1)
    mask = ImageClip(arr_mask, is_mask=True).with_duration(1)
    return overlay.with_mask(mask)


def main():
    # Load assets
    audio = AudioFileClip(AUDIO_PATH)
    duration = audio.duration

    # Lyrics + timings -> paragraphs with times
    timing_lines = load_timing_lines(TIMINGS_PATH)
    lyric_paras = load_lyric_paragraphs(LYRICS_PATH)
    paras = map_paragraph_times(lyric_paras, timing_lines)

    # Left pane clip
    left_fn = render_left_frame_factory(paras)
    left_clip = (
        VideoClip()
        .with_updated_frame_function(left_fn)
        .with_duration(duration)
        .with_fps(FPS)
    )

    # Right pane: prepare per-block rendered lines (no scrolling; line-by-line reveal)
    plan_blocks_lines = [parse_block_lines(b) for b in PLAN_BLOCKS]

    def render_plan_lines(lines: List[Tuple[str, str]]):
        rendered = []  # list of (fr, mk)
        for text, tag in lines:
            color = FG
            if tag == "error":
                color = ERR
            elif tag == "bliss":
                color = BLISS
            tc = safe_text(text or " ", MONO, PLAN_FONT_SIZE, color, method="label", text_align="left")
            fr = tc.with_duration(1 / FPS).get_frame(0)
            mk = tc.to_mask().with_duration(1 / FPS).get_frame(0)
            rendered.append((fr, mk))
        return rendered

    plan_blocks_rendered = [render_plan_lines(lines) for lines in plan_blocks_lines]

    # Per-block durations: reveal lines at a fixed rate, then pause, then clear
    line_interval = 1.0 / PLAN_LINES_PER_SEC
    offsets = []
    acc = 0.0
    block_durations = []  # total per-block (reveal + pause + clear)
    reveal_durations = [] # just the reveal duration per block
    for block in plan_blocks_rendered:
        t_reveal = len(block) * line_interval
        t_pause = PLAN_CHUNK_PAUSE
        t_clear = PLAN_CLEAR_PAUSE
        total = t_reveal + t_pause + t_clear
        offsets.append(acc)
        reveal_durations.append(t_reveal)
        block_durations.append(total)
        acc += total
    sequence_total = acc

    # Delay/Restart: use verse start times to gate cycles
    try:
        with open(TIMINGS_PATH, "r", encoding="utf-8") as _f:
            _timedata = json.load(_f)
        verse_starts = sorted(
            [float(b.get("start", 0.0)) for b in _timedata.get("blocks", []) if b.get("type") == "verse"]
        )
    except Exception:
        verse_starts = []
    # Fallback single delay if no verse found
    first_lyric_start = min((tl.start for tl in timing_lines), default=0.0)
    plan_start_delay = max(0.0, first_lyric_start + 0.05)

    def plan_frame(t: float) -> np.ndarray:
        im = np.zeros((H, RIGHT_W, 3), dtype=np.uint8)
        im[:, :, :] = BG

        if sequence_total <= 0 or not plan_blocks_rendered:
            return im

        # Determine the gating start time (per verse) and local time within the current cycle
        if verse_starts:
            vs = None
            for s in verse_starts:
                if s <= t:
                    vs = s
                else:
                    break
            if vs is None:
                return im
            tv = t - vs
            if tv >= sequence_total:
                return im  # wait for next verse start to restart sequence
        else:
            # Fallback: single delayed start, one run only
            tv = t - plan_start_delay
            if tv < 0 or tv >= sequence_total:
                return im

        # find the active chunk index by sequential offsets
        idx = 0
        for i, off in enumerate(offsets):
            if tv >= off:
                idx = i
            else:
                break
        block = plan_blocks_rendered[idx]
        t_local = tv - offsets[idx]
        t_reveal = reveal_durations[idx]
        t_pause = PLAN_CHUNK_PAUSE
        # local helper for alpha blit with clipping
        def r_alpha_blit(dest, fr, mk, x, y):
            Hd, Wd = dest.shape[:2]
            h, w = fr.shape[:2]
            if x >= Wd or y >= Hd:
                return
            if y < 0:
                cut = -y
                h -= cut
                y = 0
                fr = fr[cut:cut+h]
                mk = mk[cut:cut+h]
            if x < 0:
                cut = -x
                w -= cut
                x = 0
                fr = fr[:, cut:cut+w]
                mk = mk[:, cut:cut+w]
            if h <= 0 or w <= 0:
                return
            h = min(h, Hd - y)
            w = min(w, Wd - x)
            if h <= 0 or w <= 0:
                return
            sub = dest[y:y+h, x:x+w, :]
            a = mk[:h, :w].astype(np.float32)[..., None]
            if a.size == 0 or a.max() <= 0:
                return
            sub[:] = (a * fr[:h, :w, :].astype(np.float32) + (1.0 - a) * sub.astype(np.float32)).clip(0,255).astype(np.uint8)

        # draw function: stack lines from top, clipping horizontally to pane width
        def draw_block(n_lines):
            y = PADDING
            for i in range(min(n_lines, len(block))):
                fr, mk = block[i]
                # clip width to available area
                w = min(fr.shape[1], RIGHT_W - 2 * PADDING)
                h = fr.shape[0]
                r_alpha_blit(im, fr[:, :w], mk[:, :w], PADDING, y)
                y += h + 8

        if t_local < t_reveal:
            # reveal phase: compute how many lines should be visible
            n = int(t_local / line_interval) + 1
            draw_block(n)
        elif t_local < t_reveal + t_pause:
            # pause: draw all lines
            draw_block(len(block))
        else:
            # clear
            pass
        return im

    right_clip = (
        VideoClip()
        .with_updated_frame_function(plan_frame)
        .with_duration(duration)
        .with_fps(FPS)
        .with_position((LEFT_W, 0))
    )

    # Background
    bg = ImageClip(np.zeros((H, W, 3), dtype=np.uint8) + np.array(BG, dtype=np.uint8)).with_duration(duration)

    # Frame overlay (static)
    overlay = frame_overlay((W, H), LEFT_W, RIGHT_W).with_duration(duration)

    # Compose
    final = (
        CompositeVideoClip(
            [bg, left_clip.with_position((0, 0)), right_clip, overlay],
            size=(W, H),
        )
        .with_audio(audio)
    )

    # Render
    final.write_videofile(
        "stillalive.mp4",
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )


if __name__ == "__main__":
    main()
