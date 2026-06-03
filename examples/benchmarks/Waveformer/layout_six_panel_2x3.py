#!/usr/bin/env python3
"""将单行 6 联 beta 灵敏度图（含右侧图例）排成 2×3 六宫格."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def relayout_horizontal_six_panel(
    src: Path,
    dst: Path,
    *,
    x_left: int = 20,
    x_separators_right: tuple[int, ...] = (542, 1064, 1587, 2110, 2632, 3155),
    legend_x0: int = 3168,
    gap_x: int = 14,
    gap_y: int = 14,
    gap_legend: int = 22,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    im = Image.open(src).convert("RGB")
    w, h = im.size
    if legend_x0 >= w:
        raise ValueError(f"legend_x0={legend_x0} >= width={w}")

    legend = im.crop((legend_x0, 0, w, h))

    xs = (x_left,) + x_separators_right
    panels: list[Image.Image] = []
    for i in range(6):
        x0, x1 = xs[i], xs[i + 1]
        panels.append(im.crop((x0, 0, x1, h)))

    cell_w = max(p.width for p in panels)
    cell_h = max(p.height for p in panels)
    grid_w = 3 * cell_w + 2 * gap_x
    grid_h = 2 * cell_h + gap_y
    out_w = grid_w + gap_legend + legend.width
    out_h = grid_h
    canvas = Image.new("RGB", (out_w, out_h), bg_color)

    def paste_cell(col: int, row: int, panel: Image.Image) -> None:
        x = col * (cell_w + gap_x)
        y = row * (cell_h + gap_y)
        dx = max(0, (cell_w - panel.width) // 2)
        dy = max(0, (cell_h - panel.height) // 2)
        canvas.paste(panel, (x + dx, y + dy))

    order = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    for panel, (c, r) in zip(panels, order):
        paste_cell(c, r, panel)

    ly = max(0, (out_h - legend.height) // 2)
    canvas.paste(legend, (grid_w + gap_legend, ly))
    canvas.save(dst, dpi=(300, 300))


def main() -> None:
    here = Path(__file__).resolve().parent
    src = here / "e50b7d1811d59118d30a948f3e9c766c.png"
    dst = here / "e50b7d1811d59118d30a948f3e9c766c_2x3.png"
    relayout_horizontal_six_panel(src, dst)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
