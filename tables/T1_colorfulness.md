# T1 — Colourfulness statistics

Colourfulness follows Hasler and Susstrunk. Images are stratified at evaluation time only, never during training. Thresholds are the 33% and 67% quantiles computed within each dataset and each region definition, so the three bins are equal in size by construction. The foreground rows use the dataset's own segmentation mask, which is what R2.2 asked for: a flower or a bird occupying a small part of the frame should not inherit the colourfulness of its background.

| dataset | region | images | mean | median | SD | range | lower tertile | upper tertile | low / mid / high |
|:---|:---|---:|---:|---:|---:|:---|---:|---:|:---|
| Flowers-102 | whole image | 6149 | 69.98 | 66.85 | 30.45 | 8.07–161.18 | 52.57 | 83.63 | 2048 / 2053 / 2048 |
| Flowers-102 | foreground | 6149 | 68.85 | 68.33 | 29.35 | 6.01–171.97 | 53.01 | 83.96 | 2048 / 2053 / 2048 |
| CUB-200 | whole image | 5794 | 37.73 | 35.37 | 18.42 | 0.00–131.65 | 27.92 | 43.63 | 1930 / 1934 / 1930 |
| CUB-200 | foreground | 5794 | 40.53 | 37.57 | 20.15 | 0.00–140.77 | 29.29 | 46.64 | 1930 / 1934 / 1930 |
