"""Render flowchart scenes with vector icons and collision checks.

This is a small, self-contained renderer for the scene JSON used by this
repository. It draws rounded boxes, vector icons, orthogonal arrows and
captions, then measures every label and refuses to export if text overlaps,
a label overflows its node, or an arrow crosses a node or a label.

    python scripts/render_flowchart.py scene.json --out figure --palette jade-violet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

PALETTES = {
    "jade-violet": {"canvas": "#ffffff", "text": "#213047", "muted": "#69788c",
                    "line": "#cbd4df", "frame": "#e5eaf1", "base": "#f1f4f8",
                    "local": "#218c85", "global": "#615ced", "third": "#b9781f",
                    "fourth": "#367fc1"},
    "ocean": {"canvas": "#ffffff", "text": "#182d42", "muted": "#62758a",
              "line": "#cbd7e2", "frame": "#e6edf2", "base": "#f2f6fa",
              "local": "#2769ac", "global": "#168782", "third": "#9566b5",
              "fourth": "#ba791e"},
    "monochrome": {"canvas": "#ffffff", "text": "#20262e", "muted": "#64707c",
                   "line": "#cbd0d7", "frame": "#e7e9ed", "base": "#f2f3f5",
                   "local": "#334155", "global": "#64748b", "third": "#474750",
                   "fourth": "#77746e"},
}
SIDES = ("top", "right", "bottom", "left")


def palette(name: str) -> dict:
    c = dict(PALETTES.get(name, PALETTES["jade-violet"]))
    c["memory"] = c["fourth"]
    c["modulation"] = c["third"]
    for role in ("local", "global", "third", "fourth", "memory", "modulation"):
        rgb = tuple(int(c[role][i:i + 2], 16) for i in (1, 3, 5))
        c[role + "_soft"] = "#" + "".join(f"{round(v * .08 + 255 * .92):02x}" for v in rgb)
    c["neutral"] = c["line"]
    c["neutral_soft"] = c["base"]
    return c


# ---------------------------------------------------------------- icons
def draw_icon(ax, cx, cy, name, color, s=11):
    """Draw a small monochrome vector glyph centred on (cx, cy)."""
    lw = 1.5
    if name == "lock":
        ax.add_patch(Polygon([[cx - s * .55, cy + s * .05], [cx - s * .55, cy + s * .6],
                              [cx + s * .55, cy + s * .6], [cx + s * .55, cy + s * .05]],
                             closed=True, fill=False, ec=color, lw=lw, zorder=6))
        ax.add_patch(Circle((cx, cy + s * .05), s * .42, fill=False, ec=color, lw=lw, zorder=6))
    elif name == "db":
        ax.add_patch(Rectangle((cx - s * .6, cy - s * .6), s * 1.2, s * 1.1,
                               fill=False, ec=color, lw=lw, zorder=6))
        ax.add_patch(matplotlib.patches.Ellipse((cx, cy + s * .5), s * 1.2, s * .5,
                                                fill=False, ec=color, lw=lw, zorder=6))
        ax.add_patch(matplotlib.patches.Arc((cx, cy - s * .6), s * 1.2, s * .5, theta1=180,
                                            theta2=360, ec=color, lw=lw, zorder=6))
    elif name == "gear":
        ax.add_patch(Circle((cx, cy), s * .45, fill=False, ec=color, lw=lw, zorder=6))
        ax.add_patch(Circle((cx, cy), s * .16, fill=False, ec=color, lw=lw, zorder=6))
        for k in range(6):
            a = k * math.pi / 3
            ax.add_patch(Rectangle((cx + math.cos(a) * s * .62 - s * .13,
                                    cy + math.sin(a) * s * .62 - s * .13),
                                   s * .26, s * .26, angle=math.degrees(a), fill=False,
                                   ec=color, lw=lw, zorder=6))
    elif name == "shield":
        ax.add_patch(Polygon([[cx, cy + s * .75], [cx + s * .65, cy + s * .45],
                              [cx + s * .55, cy - s * .45], [cx, cy - s * .8],
                              [cx - s * .55, cy - s * .45], [cx - s * .65, cy + s * .45]],
                             closed=True, fill=False, ec=color, lw=lw, zorder=6))
    elif name == "image":
        ax.add_patch(Rectangle((cx - s * .75, cy - s * .55), s * 1.5, s * 1.1,
                               fill=False, ec=color, lw=lw, zorder=6))
        ax.plot([cx - s * .6, cx - s * .15, cx + s * .1], [cy - s * .45, cy + s * .05, cy - s * .25],
                color=color, lw=lw, zorder=6)
        ax.add_patch(Circle((cx + s * .38, cy + s * .28), s * .14, color=color, zorder=6))
    elif name == "chip":
        ax.add_patch(Rectangle((cx - s * .45, cy - s * .45), s * .9, s * .9,
                               fill=False, ec=color, lw=lw, zorder=6))
        for k in (-1, 1):
            for t in (-.22, .22):
                ax.plot([cx + k * s * .45, cx + k * s * .8], [cy + t * s, cy + t * s],
                        color=color, lw=lw, zorder=6)
                ax.plot([cx + t * s, cx + t * s], [cy + k * s * .45, cy + k * s * .8],
                        color=color, lw=lw, zorder=6)
    elif name == "flow":
        ax.plot([cx - s * .8, cx + s * .55], [cy, cy], color=color, lw=lw, zorder=6)
        ax.plot([cx - s * .8, cx - s * .45], [cy + s * .45, cy], color=color, lw=lw, zorder=6)
        ax.plot([cx - s * .8, cx - s * .45], [cy - s * .45, cy], color=color, lw=lw, zorder=6)
        ax.plot([cx + s * .8, cx + s * .45], [cy + s * .45, cy], color=color, lw=lw, zorder=6)
        ax.plot([cx + s * .8, cx + s * .45], [cy - s * .45, cy], color=color, lw=lw, zorder=6)
    elif name == "check":
        ax.add_patch(Circle((cx, cy), s * .85, fill=False, ec=color, lw=lw, zorder=6))
        ax.plot([cx - s * .55, cx - s * .12, cx + s * .55], [cy + s * .02, cy - s * .42, cy + s * .46],
                color=color, lw=lw * 1.6, zorder=6, solid_capstyle="round")
    elif name == "warn":
        ax.add_patch(Polygon([[cx, cy - s * .75], [cx + s * .8, cy + s * .65],
                              [cx - s * .8, cy + s * .65]], closed=True,
                             fill=False, ec=color, lw=lw, zorder=6))
        ax.plot([cx, cx], [cy - s * .2, cy + s * .3], color=color, lw=lw, zorder=6)
        ax.add_patch(Circle((cx, cy - s * .45), s * .07, color=color, zorder=6))
    elif name == "scale":
        ax.plot([cx, cx], [cy + s * .8, cy - s * .5], color=color, lw=lw, zorder=6)
        ax.plot([cx - s * .75, cx + s * .75], [cy + s * .62, cy + s * .62], color=color, lw=lw, zorder=6)
        for side in (-1, 1):
            ax.plot([cx, cx + side * s * .7], [cy + s * .62, cy + s * .18],
                    color=color, lw=lw * .8, zorder=6)
            ax.add_patch(matplotlib.patches.Arc((cx + side * s * .7, cy + s * .18), s * .55, s * .38,
                                                theta1=180, theta2=360, ec=color, lw=lw, zorder=6))
        ax.plot([cx - s * .5, cx + s * .5], [cy - s * .5, cy - s * .5], color=color, lw=lw, zorder=6)
    elif name == "arrow":
        ax.annotate("", xy=(cx + s * .8, cy), xytext=(cx - s * .8, cy),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw,
                                "mutation_scale": 10}, zorder=6)
    elif name == "layers":
        for k, off in enumerate((-.5, 0, .5)):
            ax.add_patch(Polygon([[cx, cy + s * .45 + off * s * .55], [cx + s * .75, cy + off * s * .55],
                                  [cx, cy - s * .45 + off * s * .55], [cx - s * .75, cy + off * s * .55]],
                                 closed=True, fill=False, ec=color, lw=lw, zorder=6 + k))


# ---------------------------------------------------------------- routing
def port(box, side, fraction=.5, stub=0):
    x, y, w, h = box
    p = {"top": (x + w * fraction, y), "bottom": (x + w * fraction, y + h),
         "left": (x, y + h * fraction), "right": (x + w, y + h * fraction)}[side]
    return (p[0], p[1] + stub) if side in ("top", "bottom") else (p[0] + stub, p[1])


def default_sides(a, b):
    ax_, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
    if abs(by - ay) >= abs(bx - ax_):
        return ("bottom", "top") if by > ay else ("top", "bottom")
    return ("right", "left") if bx > ax_ else ("left", "right")


def build_route(p, fs, q, ts, waypoints=None):
    """Orthogonal path between two ports, honouring the exit/entry side."""
    if waypoints:
        pts = [p] + [tuple(w) for w in waypoints] + [q]
    elif fs in ("left", "right"):
        mx = (p[0] + q[0]) / 2
        pts = [p, (mx, p[1]), (mx, q[1]), q]
    elif fs in ("top", "bottom"):
        my = (p[1] + q[1]) / 2
        pts = [p, (p[0], my), (q[0], my), q]
    else:
        pts = [p, (q[0], p[1]), q]
    out = []
    for pt in pts:
        if not out:
            out.append(pt)
            continue
        if abs(pt[0] - out[-1][0]) < 1e-6 and abs(pt[1] - out[-1][1]) < 1e-6:
            continue
        if len(out) >= 1 and (abs(out[-1][0] - pt[0]) < 1e-6 or abs(out[-1][1] - pt[1]) < 1e-6):
            out.append(pt)
        else:  # diagonal input: insert an elbow
            out.extend([(out[-1][0], pt[1]), pt])
    clean = []
    for pt in out:
        if len(clean) >= 2:
            a, b = clean[-2], clean[-1]
            if (abs(a[0] - b[0]) < 1e-6 and abs(b[0] - pt[0]) < 1e-6) or \
               (abs(a[1] - b[1]) < 1e-6 and abs(b[1] - pt[1]) < 1e-6):
                clean[-1] = pt
                continue
        clean.append(pt)
    return clean


def seg_hits_box(a, b, box, pad=4):
    x, y, w, h = box[0] - pad, box[1] - pad, box[2] + 2 * pad, box[3] + 2 * pad
    if abs(a[0] - b[0]) < 1e-6:
        return x + 0.5 < a[0] < x + w - 0.5 and max(min(a[1], b[1]), y + 0.5) < min(max(a[1], b[1]), y + h - 0.5)
    if abs(a[1] - b[1]) < 1e-6:
        return y + 0.5 < a[1] < y + h - 0.5 and max(min(a[0], b[0]), x + 0.5) < min(max(a[0], b[0]), x + w - 0.5)
    return False


# ---------------------------------------------------------------- render
def render(scene: dict, out: str, palette_name: str = "jade-violet") -> dict:
    c = palette(palette_name)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = scene["canvas"]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=c["canvas"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    texts = []
    nodes = {n["id"]: n for n in scene["nodes"]}

    def text(x, y, s, size=22, color="text", ha="center", bold=False, owner=None):
        t = ax.text(x, y, s, fontsize=size * .72, fontfamily="Noto Sans CJK JP",
                    color=c.get(color, color), ha=ha, va="center",
                    fontweight="bold" if bold else "normal", linespacing=1.35, zorder=7)
        texts.append((t, owner))

    for f in scene.get("frames", []):
        x, y, bw, bh = f["box"]
        col = c[f.get("role", "frame")]
        fill = c[f["fill"]] if f.get("fill") else "none"
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle=f"round,pad=0,rounding_size={f.get('radius', 14)}",
                                    facecolor=fill, edgecolor=col, lw=1, ls="--", zorder=0))

    for item in scene.get("lines", []):
        ax.plot(*zip(*item["points"]), color=c[item.get("role", "frame")],
                lw=item.get("width", 1), ls="--" if item.get("dashed") else "-", zorder=0)

    for n in nodes.values():
        x, y, nw, nh = n["box"]
        role = n.get("role", "neutral")
        fill = c[role] if n.get("emphasis") else c[role + "_soft"]
        if n.get("shape") == "circle":
            ax.add_patch(Circle((x + nw / 2, y + nh / 2), nw / 2, facecolor=fill,
                                edgecolor=c[role], lw=1.2, zorder=2))
        else:
            ax.add_patch(FancyBboxPatch((x, y), nw, nh, boxstyle="round,pad=0,rounding_size=10",
                                        facecolor=fill, edgecolor=c[role], lw=1.3, zorder=2))
        icon = n.get("icon")
        shift = 0
        if icon and n.get("shape") != "circle":
            ic = "#ffffff" if n.get("emphasis") else c[role] if role != "neutral" else c["muted"]
            draw_icon(ax, x + 30, y + nh / 2, icon, ic, n.get("icon_size", 11))
            shift = 20
        col = "#ffffff" if n.get("emphasis") else (role if role != "neutral" else "text")
        if n.get("sublabel"):
            text(x + nw / 2 + shift, y + nh / 2 - 13, n["label"], n.get("fontsize", 22), col,
                 bold=n.get("bold", False), owner=n["id"])
            text(x + nw / 2 + shift, y + nh / 2 + 17, n["sublabel"], n.get("subsize", 17),
                 "#ffffff" if n.get("emphasis") else "muted", owner=n["id"])
        else:
            text(x + nw / 2 + shift, y + nh / 2, n["label"], n.get("fontsize", 22), col,
                 bold=n.get("bold", False), owner=n["id"])

    for t in scene.get("captions", []):
        text(t["x"], t["y"], t["text"], t.get("size", 22), t.get("color", "text"),
             t.get("ha", "center"), t.get("bold", False))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = []
    for t, owner in texts:
        b = t.get_window_extent(renderer)
        a1, a2 = ax.transData.inverted().transform([[b.x0, b.y0], [b.x1, b.y1]])
        labels.append({"text": t.get_text(), "owner": owner,
                       "box": [min(a1[0], a2[0]), min(a1[1], a2[1]),
                               abs(a2[0] - a1[0]), abs(a2[1] - a1[1])]})

    problems = []
    for l in labels:
        x, y, lw_, lh = l["box"]
        if x < 0 or y < 0 or x + lw_ > w or y + lh > h:
            problems.append(f"page overflow: {l['text']}")
        if l["owner"]:
            nx, ny, nw, nh = nodes[l["owner"]]["box"]
            if x < nx + 4 or x + lw_ > nx + nw - 4 or y < ny + 2 or y + lh > ny + nh - 2:
                problems.append(f"node label overflow: {l['text']}")
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            x, y, bw, bh = a["box"]
            u, v, cw, ch = b["box"]
            if min(x + bw, u + cw) - max(x, u) > 1 and min(y + bh, v + ch) - max(y, v) > 1:
                problems.append(f"text overlap: {a['text']} / {b['text']}")

    drawn = []
    for e in scene["edges"]:
        a, b = nodes[e["from"]], nodes[e["to"]]
        fs, ts = (e.get("from_side"), e.get("to_side"))
        if fs is None or ts is None:
            dfs, dts = default_sides(a["box"], b["box"])
            fs = fs or dfs
            ts = ts or dts
        p = port(a["box"], fs, e.get("from_fraction", .5))
        q = port(b["box"], ts, e.get("to_fraction", .5))
        pts = build_route(p, fs, q, ts, e.get("waypoints"))
        for u, v in zip(pts, pts[1:]):
            for n in nodes.values():
                if n["id"] in (a["id"], b["id"]):
                    continue
                if seg_hits_box(u, v, n["box"]):
                    problems.append(f"edge {a['id']}->{b['id']} crosses node {n['id']}")
            for l in labels:
                if l["owner"] in (a["id"], b["id"]):
                    continue
                if seg_hits_box(u, v, l["box"], pad=3):
                    problems.append(f"edge {a['id']}->{b['id']} crosses text '{l['text']}'")
        drawn.append((e, pts))

    if problems:
        raise SystemExit("\n".join(sorted(set(problems))))

    for e, pts in drawn:
        col = c[e.get("role", "muted")]
        weight = e.get("width", 1.5)
        ax.plot(*zip(*pts), color=col, lw=weight, ls="--" if e.get("dashed") else "-",
                zorder=3, solid_capstyle="round")
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=13,
                                     color=col, lw=weight, zorder=4, shrinkA=0, shrinkB=0))

    for ext in ("png", "svg"):
        fig.savefig(str(out) + "." + ext, dpi=160, metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)

    report = {"palette": palette_name, "nodes": len(nodes), "edges": len(drawn),
              "text_overlaps": 0, "node_label_overflow": 0,
              "arrow_node_crossings": 0, "arrow_text_crossings": 0, "status": "pass"}
    Path(str(out) + ".validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene")
    ap.add_argument("--out", required=True)
    ap.add_argument("--palette", default="jade-violet")
    args = ap.parse_args()
    scene = json.loads(Path(args.scene).read_text())
    report = render(scene, args.out, args.palette)
    report["scene_sha256"] = hashlib.sha256(Path(args.scene).read_bytes()).hexdigest()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
