#!/usr/bin/env python
"""Rewrite the "Raw Partition" IIA labels of Figures 3 and 4 to directed IIA.

Background
----------
Figures 3 and 4 each used the name "IIA" for two different quantities: the
classifier rows carry *directed* IIA (correct ordered pairs / n(n-1)) while the
"Raw Partition" rows carried *undirected density* (an edge requires the
interchange intervention to succeed in both directions).  The paper
standardises on directed IIA, so only the Raw Partition rows change.  All seven
replacement values are produced by ``experiments/recompute_directed_iia.py``;
run that first.

Why this edits pixels instead of re-plotting
--------------------------------------------
The original plotting code for Figures 3 and 4 is not in this repository, and
the SAE-classifier rows cannot be redrawn at all: their per-input bucket
memberships were never saved (the jsons keep only ``n`` and the metrics), so
rebuilding a whole panel would require re-running the SAE classifier.  Those
rows are already directed and correct.  Bar *lengths* are counts and do not
change either.  So the minimal correct edit is to replace the individual digit
glyphs of the Raw Partition IIA labels, leaving every other pixel untouched:

  * Figure 4's Diagnosis panel is an embedded raster inside
    ``entity_binding_new.svg``.  Its text is a plain matplotlib render
    (DejaVu Sans Bold, 16.0pt for the y-labels and 18.0pt for the panel title,
    at 100 dpi), which this script reproduces and verifies against the original
    before swapping single glyphs (alignment RMSE < 2/255).  The panel is
    re-embedded losslessly as PNG so that only the edited glyph pixels differ
    from the previously embedded JPEG's decoded content.
  * ``logic2_new.png`` has no SVG and its panels were resampled on the way into
    the composed figure, so a fresh render does not match.  Its glyphs are
    therefore copied from other instances of the same digit elsewhere in the
    *same* figure, sub-pixel shifted into place.  The shift is measured against
    ground truth (the shared ``IIA: 0.`` run of the source and target lines, or
    a digit-cell pitch calibrated on lines that repeat a digit in adjacent
    cells), so the copied glyph carries exactly the surrounding text's
    rasterisation.
  * Both full-resolution composed PNGs are patched with a *delta*: the change
    is added to the published pixel values, so every pixel outside an edited
    glyph stays byte-identical.

Usage
-----
    python experiments/redraw_directed_iia_figures.py            # edit in place
    python experiments/redraw_directed_iia_figures.py --outdir /tmp/out

The script refuses to run unless the inputs are the pre-edit originals (it
checks their SHA-256), so it cannot be applied twice.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "ArXiv", "figs")

# SHA-256 of the pre-edit figures (commit 18a7f4a).
SRC_SHA = {
    "logic2_new.png":
        "fb1cb5e62a829114d7ffd127c6c0173e9b2dc368d42000b9d1643515f0a842ae",
    "entity_binding_new.png":
        "4fea00f67fa8ef04e9d1984509418e70d4736e6e16a690ebaffc19ee8b453161",
    "entity_binding_new.svg":
        "1abf2793031179acf3be54731c32e80c5f32e389ff50f432cc3fbe29cda11ede",
}


# --------------------------------------------------------------------------
# separable resampling with an explicit continuous source origin
# --------------------------------------------------------------------------
def _kern(t, name):
    t = np.abs(t)
    if name == "tent":
        return np.clip(1.0 - t, 0, None)
    if name == "cubic":                      # Catmull-Rom (B=0, C=1/2)
        t2, t3 = t * t, t * t * t
        w = np.zeros_like(t)
        m1, m2 = t < 1, (t >= 1) & (t < 2)
        w[m1] = 1.5 * t3[m1] - 2.5 * t2[m1] + 1.0
        w[m2] = -0.5 * t3[m2] + 2.5 * t2[m2] - 4.0 * t[m2] + 2.0
        return w
    raise ValueError(name)


_SUPPORT = {"tent": 1.0, "cubic": 2.0}


def _axis_w(n_out, X0, S, n_in, name):
    """Weights for out px X, whose centre samples the source at ((X+.5)-X0)/S-.5."""
    scale = 1.0 if S >= 1.0 else S
    support = _SUPPORT[name] / scale
    X = np.arange(n_out)
    u = ((X + 0.5) - X0) / S - 0.5
    lo = np.floor(u - support).astype(int) + 1
    k = int(np.ceil(2 * support)) + 1
    idx = lo[:, None] + np.arange(k)[None, :]
    w = _kern((u[:, None] - idx) * scale, name)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return np.clip(idx, 0, n_in - 1), w / s


def resample(a, out_w, out_h, X0, Y0, Sx, Sy, name="tent"):
    a = np.asarray(a, dtype=np.float64)
    ix, wx = _axis_w(out_w, X0, Sx, a.shape[1], name)
    iy, wy = _axis_w(out_h, Y0, Sy, a.shape[0], name)
    tmp = np.einsum("yxk,xk->yx", a[:, ix], wx)
    return np.einsum("ykx,yk->yx", tmp[iy], wy)


def sample_at(A, x0f, y0f, w, h, name="cubic"):
    """Sample A on an integer grid whose origin sits at continuous (x0f, y0f)."""
    pad = 4
    ix0, iy0 = int(np.floor(x0f)) - pad, int(np.floor(y0f)) - pad
    sub = A[iy0:iy0 + h + 2 * pad + 2, ix0:ix0 + w + 2 * pad + 2]
    return resample(sub, w, h, -(x0f - ix0), -(y0f - iy0), 1.0, 1.0, name)


def fit_shift(A, sx, sy, tx, ty, w, h, name="cubic", rng=2.0, iters=6):
    """Continuous source origin near (sx,sy) best reproducing A[ty:ty+h, tx:tx+w]."""
    tgt = A[ty:ty + h, tx:tx + w].astype(float)
    cx, cy, step, best = float(sx), float(sy), 0.5, None
    for _ in range(iters):
        cand = []
        for dx in np.arange(-rng, rng + 1e-9, step):
            for dy in np.arange(-rng, rng + 1e-9, step):
                out = sample_at(A, cx + dx, cy + dy, w, h, name)
                d = out - tgt
                cand.append((float(np.sqrt((d * d).mean())), cx + dx, cy + dy))
        cand.sort()
        best = cand[0]
        cx, cy = best[1], best[2]
        rng, step = step, step / 2
    return best


# --------------------------------------------------------------------------
# Figure 4 -- matplotlib text re-render, single-glyph swap
# --------------------------------------------------------------------------
def render_canvas(text, fontsize, w=900, h=80, penx=20, peny=40, dpi=100):
    """Render `text` at a fixed pen position, so two strings are pixel-comparable."""
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.text(penx / w, 1.0 - peny / h, text, fontsize=fontsize, fontweight="bold",
             family="DejaVu Sans", color="black", ha="left", va="baseline")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].mean(axis=2).astype(float)
    plt.close(fig)
    return buf


def _align(tpl, big, xrange_, yrange_):
    ph, pw = tpl.shape
    best = None
    for dy in range(yrange_[0], yrange_[1] - ph + 1):
        for dx in range(xrange_[0], xrange_[1] - pw + 1):
            d = big[dy:dy + ph, dx:dx + pw] - tpl
            r = float(np.sqrt((d * d).mean()))
            if best is None or r < best[0]:
                best = (r, dx, dy)
    return best


def patch_glyph_text(img_rgb, old, new, fontsize, xrange_, yrange_, label,
                     strong=40.0, margin=2, max_rmse=3.0):
    """Swap the differing glyph of `old`->`new` inside img_rgb (uint8 HxWx3)."""
    Rold, Rnew = render_canvas(old, fontsize), render_canvas(new, fontsize)
    ys, xs = np.where(Rold < 250)
    ox0, oy0, ox1, oy1 = xs.min(), ys.min(), xs.max(), ys.max()
    tpl = Rold[oy0 - margin:oy1 + margin + 1, ox0 - margin:ox1 + margin + 1]

    gray = np.asarray(Image.fromarray(img_rgb).convert("L")).astype(float)
    rmse, X, Y = _align(tpl, gray, xrange_, yrange_)
    if rmse > max_rmse:
        raise AssertionError(
            "%s: could not reproduce the original text (alignment RMSE %.2f > %.2f); "
            "the font/size assumption no longer holds" % (label, rmse, max_rmse))
    offx, offy = X - (ox0 - margin), Y - (oy0 - margin)

    D = np.abs(Rold - Rnew)
    sy, sx = np.where(D >= strong)          # the one glyph that really changed
    px0, px1 = sx.min() - margin, sx.max() + margin
    py0, py1 = sy.min() - margin, sy.max() + margin

    out = img_rgb.copy()
    src = np.clip(np.round(Rnew[py0:py1 + 1, px0:px1 + 1]), 0, 255).astype(np.uint8)
    ix0, iy0 = px0 + offx, py0 + offy
    ih, iw = src.shape
    out[iy0:iy0 + ih, ix0:ix0 + iw] = np.repeat(src[:, :, None], 3, axis=2)
    info = dict(label=label, old=old, new=new, fontsize=fontsize,
                align_rmse=round(rmse, 2),
                panel_rect=[int(ix0), int(iy0), int(ix0 + iw - 1), int(iy0 + ih - 1)])
    print("  %-26s %-20s -> %-20s align_rmse=%.2f  panel rect x=%d..%d y=%d..%d"
          % (label, old, new, rmse, ix0, ix0 + iw - 1, iy0, iy0 + ih - 1))
    return out, info


# Geometry of the Diagnosis panel inside entity_binding_new.png, fitted to
# RMSE 0.57/255 against the published raster (plain bilinear, 300 dpi export).
EB_X0, EB_Y0, EB_SX, EB_SY = 2693.78125, 165.625, 1.32961, 1.327795

FIG4_EDITS = [
    ("Overall IIA = 0.44", "Overall IIA = 0.48", 18.0, (500, 830), (30, 64),
     "raw-partition title"),
    ("(n=129, IIA=0.98)", "(n=129, IIA=0.99)", 16.0, (4, 238), (140, 170),
     "target bucket n=129"),
    ("(n=378, IIA=0.11)", "(n=378, IIA=0.31)", 16.0, (4, 238), (270, 300),
     "other bucket n=378"),
]


def do_figure4(svg_path, png_path, outdir):
    svg = open(svg_path, encoding="utf-8").read()
    m = re.search(r'<image width="1187" height="1167" xlink:href="data:image/jpeg;'
                  r'base64,([^"]+)" preserveAspectRatio="none" id="img2"></image>', svg)
    assert m, "Diagnosis <image id=img2> not found in the SVG"
    panel0 = np.asarray(Image.open(io.BytesIO(base64.b64decode(m.group(1))))
                        .convert("RGB")).copy()

    print("Figure 4 - Diagnosis panel (%dx%d) glyph swaps:" % panel0.shape[1::-1])
    panel = panel0.copy()
    infos = []
    for old, new, fs, xr, yr, lab in FIG4_EDITS:
        panel, info = patch_glyph_text(panel, old, new, fs, xr, yr, lab)
        infos.append(info)
    changed = int((panel != panel0).any(axis=2).sum())
    print("  panel pixels changed: %d of %d" % (changed, panel0.shape[0] * panel0.shape[1]))

    # --- swap the one <image> in the SVG, losslessly
    buf = io.BytesIO()
    Image.fromarray(panel).save(buf, format="PNG", optimize=True)
    new_elem = ('<image width="1187" height="1167" xlink:href="data:image/png;base64,'
                + base64.b64encode(buf.getvalue()).decode("ascii")
                + '" preserveAspectRatio="none" id="img2"></image>')
    svg_out = svg.replace(m.group(0), new_elem)
    pre_o, post_o = svg.split(m.group(0))
    pre_n, post_n = svg_out.split(new_elem)
    assert pre_o == pre_n and post_o == post_n, "SVG changed outside the img2 element"
    open(os.path.join(outdir, "entity_binding_new.svg"), "w", encoding="utf-8").write(svg_out)
    print("  SVG: swapped img2 only; %d -> %d bytes" % (len(svg.encode()), len(svg_out.encode())))

    # --- delta-patch the composed 300 dpi PNG
    big = np.asarray(Image.open(png_path)).astype(np.int32).copy()
    o, n = panel0.astype(float), panel.astype(float)
    total = 0
    for info in infos:
        x0, y0, x1, y1 = info["panel_rect"]
        PX0 = int(np.floor(EB_X0 + (x0 - 2 + 0.5) * EB_SX - 0.5))
        PX1 = int(np.ceil(EB_X0 + (x1 + 3 + 0.5) * EB_SX - 0.5))
        PY0 = int(np.floor(EB_Y0 + (y0 - 2 + 0.5) * EB_SY - 0.5))
        PY1 = int(np.ceil(EB_Y0 + (y1 + 3 + 0.5) * EB_SY - 0.5))
        w, h = PX1 - PX0, PY1 - PY0
        ro = np.dstack([resample(o[:, :, c], w, h, EB_X0 - PX0, EB_Y0 - PY0,
                                 EB_SX, EB_SY, "tent") for c in range(3)])
        rn = np.dstack([resample(n[:, :, c], w, h, EB_X0 - PX0, EB_Y0 - PY0,
                                 EB_SX, EB_SY, "tent") for c in range(3)])
        delta = rn - ro
        mask = (np.abs(delta) > 0.5).any(axis=2)
        act = big[PY0:PY1, PX0:PX1, :3].astype(float)
        tgt = big[PY0:PY1, PX0:PX1, :3]
        tgt[mask] = np.clip(np.round(act + delta), 0, 255)[mask].astype(np.int32)
        ys, xs = np.where(mask)
        info["png_rect"] = [int(PX0 + xs.min()), int(PY0 + ys.min()),
                            int(PX0 + xs.max()), int(PY0 + ys.max())]
        info["png_px_changed"] = int(mask.sum())
        total += int(mask.sum())
        print("  PNG %-26s rect x=%d..%d y=%d..%d  px=%d"
              % (info["label"], info["png_rect"][0], info["png_rect"][2],
                 info["png_rect"][1], info["png_rect"][3], mask.sum()))
    Image.fromarray(big.astype(np.uint8), "RGBA").save(
        os.path.join(outdir, "entity_binding_new.png"),
        dpi=(299.9994, 299.9994), optimize=True)
    print("  PNG pixels changed: %d of %d" % (total, big.shape[0] * big.shape[1]))
    return infos


# --------------------------------------------------------------------------
# Figure 3 -- copy a digit glyph from elsewhere in the same figure
# --------------------------------------------------------------------------
# Digit-cell ink ranges and the shared "IIA: 0." run, per right-hand panel.
L2_PANEL = {
    "o5": dict(ctx=(2382, 60),
               lines=dict(raw_t=275, raw_o=337, ft_t=450, ft_o=512,
                          sae_t=626, sae_o=687)),
    "o4": dict(ctx=(2396, 62),
               lines=dict(raw_t=1117, raw_o=1179, ft_t=1292, ft_o=1354,
                          sae_t=1469, sae_o=1530)),
}
L2_PITCH = 12.583      # digit-cell pitch, mean of the '7'->'7' and '0'->'0' twins
L2_WH, L2_DY0, L2_KERN = 22, -3, "cubic"

# (panel, target line, analysis window, source line, mode, label)
# The window is chosen so the *source* read region stays clear of the source
# glyph's neighbours.
FIG3_EDITS = [
    ("o5", "raw_t", (2454, 2470), "raw_o", "same_cell",
     "o5 raw target  0.98->0.99  hundredths '8'->'9'"),
    ("o5", "raw_o", (2442, 2454), "raw_t", "same_cell",
     "o5 raw other   0.89->0.95  tenths '8'->'9'"),
    ("o5", "raw_o", (2454, 2470), "sae_t", "same_cell",
     "o5 raw other   0.89->0.95  hundredths '9'->'5'"),
    ("o4", "raw_t", (2470, 2483), "raw_t", "cell_shift",
     "o4 raw target  0.98->0.99  hundredths '8'->'9'"),
    ("o4", "raw_o", (2457, 2470), "sae_o", "same_cell",
     "o4 raw other   0.68->0.78  tenths '6'->'7'"),
]


def do_figure3(png_path, outdir, seam_tol=4.0):
    im = Image.open(png_path)
    rgba = np.asarray(im).astype(np.int32).copy()
    orig = np.asarray(im).astype(np.int32)
    pristine = np.asarray(im.convert("L")).astype(np.float64)   # sources read here

    print("Figure 3 - logic2_new.png glyph copies:")
    infos = []
    for panel, tgt_line, win, src_line, mode, label in FIG3_EDITS:
        P = L2_PANEL[panel]
        ty0 = P["lines"][tgt_line] + L2_DY0
        if mode == "same_cell":
            cx, cw = P["ctx"]
            rmse, sxf, syf = fit_shift(pristine, cx, P["lines"][src_line] + L2_DY0,
                                       cx, ty0, cw, L2_WH, L2_KERN)
            dx, dy = sxf - cx, syf - ty0
        else:                                   # shift one digit cell to the right
            rmse, dx, dy = 0.0, -L2_PITCH, 0.0
        x0, x1 = win
        w = x1 - x0 + 1
        patch = sample_at(pristine, x0 + dx, ty0 + dy, w, L2_WH, L2_KERN)
        old = pristine[ty0:ty0 + L2_WH, x0:x0 + w]
        delta = np.abs(patch - old)
        ys, xs = np.where(delta > 12.0)
        px0, px1 = max(xs.min() - 1, 0), min(xs.max() + 1, w - 1)
        py0, py1 = max(ys.min() - 1, 0), min(ys.max() + 1, L2_WH - 1)
        outside = delta.copy()
        outside[py0:py1 + 1, px0:px1 + 1] = 0.0
        if outside.max() > seam_tol:
            raise AssertionError("%s: pasted rect would leave a seam (max |delta| "
                                 "outside rect = %.1f > %.1f)"
                                 % (label, outside.max(), seam_tol))
        sub = np.clip(np.round(patch[py0:py1 + 1, px0:px1 + 1]), 0, 255).astype(np.int32)
        ax0, ay0 = x0 + px0, ty0 + py0
        h_, w_ = sub.shape
        before = rgba[ay0:ay0 + h_, ax0:ax0 + w_, :3].copy()
        for c in range(3):
            rgba[ay0:ay0 + h_, ax0:ax0 + w_, c] = sub
        nch = int((rgba[ay0:ay0 + h_, ax0:ax0 + w_, :3] != before).any(axis=2).sum())
        infos.append(dict(label=label, mode=mode, source_line=src_line,
                          ctx_fit_rmse=round(rmse, 2),
                          shift=[round(dx, 3), round(dy, 3)],
                          rect=[int(ax0), int(ay0), int(ax0 + w_ - 1), int(ay0 + h_ - 1)],
                          px_changed=nch,
                          max_delta_outside_rect=round(float(outside.max()), 1)))
        print("  %-46s src=%-6s shift=(%+.3f,%+.3f) rect x=%d..%d y=%d..%d px=%3d seam=%.1f"
              % (label, src_line, dx, dy, ax0, ax0 + w_ - 1, ay0, ay0 + h_ - 1,
                 nch, outside.max()))

    assert (rgba[:, :, 3] == orig[:, :, 3]).all(), "alpha channel must not change"
    d = (rgba != orig).any(axis=2)
    print("  PNG pixels changed: %d of %d" % (d.sum(), rgba.shape[0] * rgba.shape[1]))
    Image.fromarray(rgba.astype(np.uint8), "RGBA").save(
        os.path.join(outdir, "logic2_new.png"), dpi=(329.9968, 329.9968), optimize=True)
    return infos


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figs", default=FIGS, help="directory holding the input figures")
    ap.add_argument("--outdir", default=None, help="where to write (default: --figs)")
    ap.add_argument("--report", default=None, help="write a JSON report here")
    args = ap.parse_args()
    outdir = args.outdir or args.figs
    os.makedirs(outdir, exist_ok=True)

    for name, want in SRC_SHA.items():
        p = os.path.join(args.figs, name)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if got != want:
            sys.exit("%s is not the pre-edit original (sha256 %s, expected %s).\n"
                     "These edits have most likely already been applied; refusing "
                     "to run so the figures are not double-patched." % (p, got[:16], want[:16]))
    print("input figures match the pre-edit originals\n")

    i4 = do_figure4(os.path.join(args.figs, "entity_binding_new.svg"),
                    os.path.join(args.figs, "entity_binding_new.png"), outdir)
    print()
    i3 = do_figure3(os.path.join(args.figs, "logic2_new.png"), outdir)

    if args.report:
        json.dump(dict(figure4=i4, figure3=i3), open(args.report, "w"), indent=1)
    print("\nwrote logic2_new.png, entity_binding_new.png, entity_binding_new.svg to %s" % outdir)


if __name__ == "__main__":
    main()
