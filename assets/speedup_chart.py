#!/usr/bin/env python3
"""Draw assets/speedup-light.svg and assets/speedup-dark.svg for the front page.

Every row is one model on one Radeon AI PRO R9700: the stock engine against the Paiton release.
The numbers are copied from the model guides (the source is named per row); edit ROWS and run this file again.
"""
from pathlib import Path

# (model, workload, stock, Paiton, unit, higher_is_better, source)
ROWS = [
    ("Qwen3.8 27B", "chat and coding, MXFP4", 4.5, 113.5, "tok/s", True, "models/Qwen3.8-MXFP4-DFlash2/BENCHMARKS.md"),
    ("Qwen-Image-2.1 Turbo", "1024 x 1024 image", 23.6, 3.99, "s", False, "models/Qwen-Image-2.1/README.md"),
    ("Idol", "one image to 3D", 39.2, 16.4, "s", False, "models/Idol/README.md"),
    ("GPT-OSS-20B", "512 tokens in, 256 out", 4.819, 2.214, "s", False, "models/GPT-OSS-20B/README.md"),
    ("MiniCPM5-2B", "one request", 127.6, 197.0, "tok/s", True, "models/MiniCPM5-2B/BENCHMARKS.md"),
    ("Ornith 1.5 35B", "chat", 35.132, 44.628, "tok/s", True, "models/Ornith-1.5/README.md"),
    ("Qwen3-Coder 30B", "one request", 104.42, 126.68, "tok/s", True, "models/Qwen3-Coder-30B/BENCHMARKS.md"),
    ("FLUX.2 klein 4B", "one image", 1.258, 1.054, "s", False, "models/FLUX.2-klein/BENCHMARKS.md"),
    ("MiniMax H3", "5 s video with audio", 64.15, 54.38, "s", False, "models/MiniMax-H3/BENCHMARKS.md"),
    ("FastWan 5B", "832 x 480 video", 9.280, 8.843, "s", False, "models/FastWan/README.md"),
    ("Wan2.2 TI2V-5B", "image to video", 29.518, 29.758, "s", False, "models/Wan2.2/README.md"),
]

THEMES = {  # bar colours: orange slot of the validated palette, one step per surface
    "light": dict(ink="#0b0b0b", muted="#52514e", bar="#eb6834", rule="#52514e"),
    "dark": dict(ink="#f0efec", muted="#c3c2b7", bar="#d95926", rule="#c3c2b7"),
}
W, LEFT, PLOT, ROW, TOP = 900, 250, 380, 34, 56


def fmt(v):
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}".rstrip("0").rstrip(".")


def draw(theme):
    c = THEMES[theme]
    rows = [(m, w, s, p, u, (p / s if up else s / p)) for m, w, s, p, u, up, _ in ROWS]
    top = max(r[5] for r in rows)
    px = lambda x: LEFT + x / top * PLOT
    height = TOP + ROW * len(rows) + 52
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" role="img" '
        f'aria-label="Bar chart: how many times faster Paiton is than the stock engine on one Radeon AI PRO R9700, per model">',
        f'<g font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" font-size="13" fill="{c["ink"]}">',
        f'<text x="0" y="18" font-size="16" font-weight="600">Paiton against the stock engine, same model on one Radeon AI PRO R9700</text>',
        f'<text x="0" y="38" fill="{c["muted"]}">Times faster than stock (bar), with the measured values. The line marks stock = 1x.</text>',
    ]
    for i, (model, work, stock, paiton, unit, speedup) in enumerate(rows):
        y = TOP + i * ROW
        x1 = px(speedup)
        out += [
            f'<text x="{LEFT - 12}" y="{y + 12}" text-anchor="end" font-weight="600">{model}</text>',
            f'<text x="{LEFT - 12}" y="{y + 26}" text-anchor="end" font-size="11" fill="{c["muted"]}">{work}</text>',
            # square at the baseline, 4 px rounded at the data end
            f'<path d="M{LEFT} {y + 4}H{x1 - 4:.1f}a4 4 0 0 1 4 4v8a4 4 0 0 1 -4 4H{LEFT}z" fill="{c["bar"]}"/>',
            f'<text x="{x1 + 8:.1f}" y="{y + 18}"><tspan font-weight="700">{speedup:.1f}x</tspan>'
            f'<tspan fill="{c["muted"]}" dx="8">{fmt(stock)} to {fmt(paiton)} {unit}</tspan></text>',
        ]
    bottom = TOP + ROW * len(rows)
    out += [
        f'<path d="M{px(1):.1f} {TOP - 4}V{bottom}" stroke="{c["rule"]}" stroke-width="1" stroke-dasharray="3 3"/>',
        f'<text x="0" y="{bottom + 24}" font-size="11" fill="{c["muted"]}">Each pair runs the same model on the same card. The top two rows '
        f'compare whole setups: Paiton adds speculative decoding (27B)</text>',
        f'<text x="0" y="{bottom + 39}" font-size="11" fill="{c["muted"]}">and a quantized model that fits in VRAM (Qwen-Image). '
        f'Dates, workloads and limits are in each model guide.</text>',
        "</g></svg>",
    ]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    here = Path(__file__).parent
    for theme in THEMES:
        (here / f"speedup-{theme}.svg").write_text(draw(theme), encoding="utf8")
    assert all(Path(here.parent, r[6]).exists() for r in ROWS), "a source file named in ROWS is missing"
    print("wrote speedup-light.svg and speedup-dark.svg")
