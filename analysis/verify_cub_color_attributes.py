# -*- coding: utf-8 -*-
"""Verifikasi klaim naskah (Conclusion): "239 of the 312 attributes name the
color of a body part" untuk CUB-200-2011.

CUB-200-2011 memberi 312 atribut biner per citra. Atribut warna berformat
`has_<bagian>_color::<warna>` (mis. `has_wing_color::blue`). Skrip ini membaca
`attributes.txt` bawaan dataset, menghitung total atribut dan atribut warna,
lalu memecahnya per bagian tubuh. Hasilnya ditulis ke
`tables/cub_color_attributes.md` supaya bisa diperiksa tanpa menjalankan ulang.

`data/` di-.gitignore (dataset diunduh otomatis), jadi `attributes.txt` hadir
setelah CUB terunduh. Skrip tidak memuat dependensi selain pustaka standar.

Jalankan: python analysis/verify_cub_color_attributes.py
"""
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CANDIDATES = [
    os.path.join(ROOT, "data", "attributes.txt"),
    os.path.join(ROOT, "data", "CUB_200_2011", "attributes", "attributes.txt"),
    os.path.join(ROOT, "data", "CUB_200_2011", "attributes.txt"),
]


def find_attributes():
    for p in CANDIDATES:
        if os.path.exists(p):
            return p
    raise SystemExit(
        "attributes.txt tidak ditemukan. Unduh CUB-200-2011 dulu; file ini ada "
        "di attributes/attributes.txt pada rilis resmi dataset."
    )


def main():
    path = find_attributes()
    lines = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
    total = len(lines)

    # atribut warna: has_<bagian>_color::<warna>
    color = [ln for ln in lines if "_color::" in ln]
    per_part = Counter(
        re.search(r"has_(\w+?)_color::", ln).group(1)
        for ln in color
        if re.search(r"has_(\w+?)_color::", ln)
    )

    # jaring pengaman: angka yang dikutip naskah
    assert total == 312, f"total atribut {total}, diharapkan 312"
    assert len(color) == 239, f"atribut warna {len(color)}, diharapkan 239"

    parts = sorted(per_part)
    out = []
    out.append("# CUB-200-2011 color-attribute count\n")
    out.append(f"Source file: `{os.path.relpath(path, ROOT).replace(os.sep, '/')}` "
               "(CUB-200-2011 official attribute list).\n")
    out.append("Supports the manuscript Conclusion: "
               "\"239 of the 312 attributes name the color of a body part.\"\n")
    out.append(f"- Total attributes: **{total}**")
    out.append(f"- Color attributes (`has_<part>_color::<value>`): "
               f"**{len(color)}** ({len(color)/total*100:.1f}%)")
    out.append(f"- Non-color attributes (shape / size / pattern): "
               f"**{total - len(color)}**")
    out.append(f"- Body parts with color attributes: **{len(parts)}**\n")
    out.append("| Body part | Color values |")
    out.append("|---|---|")
    for p in parts:
        out.append(f"| {p} | {per_part[p]} |")
    out.append(f"| **total** | **{sum(per_part.values())}** |\n")
    note = ("Note: 15 parts carry 15 color values each and `eye` carries 14 "
            f"(15x15 + 14 = {15*15+14}), so the count is 239, not 16x15=240.")
    out.append(note)
    text = "\n".join(out) + "\n"

    tables = os.path.join(ROOT, "tables")
    os.makedirs(tables, exist_ok=True)
    dst = os.path.join(tables, "cub_color_attributes.md")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print(f"-> wrote {os.path.relpath(dst, ROOT)}")


if __name__ == "__main__":
    main()
