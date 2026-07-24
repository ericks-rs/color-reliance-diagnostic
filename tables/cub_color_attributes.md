# CUB-200-2011 color-attribute count

Source file: `data/attributes.txt` (CUB-200-2011 official attribute list).

Supports the manuscript Conclusion: "239 of the 312 attributes name the color of a body part."

- Total attributes: **312**
- Color attributes (`has_<part>_color::<value>`): **239** (76.6%)
- Non-color attributes (shape / size / pattern): **73**
- Body parts with color attributes: **16**

| Body part | Color values |
|---|---|
| back | 15 |
| belly | 15 |
| bill | 15 |
| breast | 15 |
| crown | 15 |
| eye | 14 |
| forehead | 15 |
| leg | 15 |
| nape | 15 |
| primary | 15 |
| throat | 15 |
| under_tail | 15 |
| underparts | 15 |
| upper_tail | 15 |
| upperparts | 15 |
| wing | 15 |
| **total** | **239** |

Note: 15 parts carry 15 color values each and `eye` carries 14 (15x15 + 14 = 239), so the count is 239, not 16x15=240.
