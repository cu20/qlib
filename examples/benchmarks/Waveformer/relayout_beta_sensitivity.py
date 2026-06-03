#!/usr/bin/env python3
"""将 beta 灵敏度图排成两行三列（2×3）。

当前 PNG 仅有四个子图（IC / ICIR / RankIC / RankICIR）；下行右侧两格留白以对应原六联图中的 AR / IR。
若你恢复完整六联位图（六段等宽子图），可将 ``n_panels`` 改为 6 并去掉留白逻辑。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def main() -> None:
    path = Path(__file__).resolve().parent / "beta_sensitivity_csi300_csi800_best_models.png"
    im = Image.open(path).convert("RGB")
    w, h = im.size

    plot_w = 2110
    legend_x0 = 2160
    legend = im.crop((legend_x0, 0, w, h))

    strip = im.crop((0, 0, plot_w, h))
    n = 4
    cw = plot_w // n
    panels = []
    for k in range(n):
        x0 = k * cw
        x1 = plot_w if k == n - 1 else (k + 1) * cw
        panels.append(strip.crop((x0, 0, x1, h)))

    gap = 14
    cell_w = max(p.width for p in panels)
    cell_h = max(p.height for p in panels)
    bg = (255, 255, 255)
    subtle = (248, 248, 248)

    grid_w = 3 * cell_w + 2 * gap
    grid_h = 2 * cell_h + gap
    gap_legend = 20
    out_w = grid_w + gap_legend + legend.width
    out_h = grid_h
    canvas = Image.new("RGB", (out_w, out_h), bg)

    def cell_origin(col: int, row: int) -> tuple[int, int]:
        return (col * (cell_w + gap), row * (cell_h + gap))

    placements = [
        (panels[0], 0, 0),
        (panels[1], 1, 0),
        (panels[2], 2, 0),
        (panels[3], 0, 1),
    ]
    for pimg, c, r in placements:
        x, y = cell_origin(c, r)
        dx = max(0, (cell_w - pimg.width) // 2)
        dy = max(0, (cell_h - pimg.height) // 2)
        canvas.paste(pimg, (x + dx, y + dy))

    for c in (1, 2):
        x, y = cell_origin(c, 1)
        patch = Image.new("RGB", (cell_w, cell_h), subtle)
        canvas.paste(patch, (x, y))

    lx = grid_w + gap_legend
    ly = max(0, (out_h - legend.height) // 2)
    canvas.paste(legend, (lx, ly))

    canvas.save(path, dpi=(300, 300))
    print(f"Saved {path} size={canvas.size}")


if __name__ == "__main__":
    main()
