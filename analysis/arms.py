"""Satu sumber kebenaran untuk arm, label, dan warna figur revisi.

KENAPA BERKAS INI ADA
Generator v1 (analyze.py, analyze_extra.py) memakai `convnext_tiny` dan
`vit_small`, yaitu checkpoint in12k_ft_in1k dan augreg_in21k_ft_in1k. Dua model
itu membawa data pretraining tambahan sementara ResNet-50 dan Swin-T hanya
ImageNet-1k. Reviewer 1 menyoroti persis hal ini (R1.2), dan keputusan revisi
sudah diambil: semua arm utama disetarakan ke IN1k, dan checkpoint non-setara
turun ke tabel confound A6 saja.

Selisihnya besar dan mengubah kesimpulan, bukan hanya angka:

    CUB clean    convnext_tiny 0.8866  vs  convnext_tiny_in1k 0.8375
    CUB clean    vit_small     0.8498  vs  vit_small_in1k     0.7951

    CR Flowers   arm kuat   -> ViT peringkat 3, ConvNeXt terendah
                 arm setara -> ViT TERTINGGI (0.5353), unggul 0.185 dari ResNet
    CR CUB       arm kuat   -> ViT peringkat 2
                 arm setara -> ViT TERTINGGI (0.4971), unggul 0.067 dari Swin

Headline revisi adalah "ViT paling color-reliant di bawah protokol bersama".
Arm setara menyangga klaim itu, arm kuat membantahnya. Jadi memakai daftar yang
salah bukan sekadar bikin angka meleset, tapi membalik temuan.

Kesalahan ini sudah terjadi dua kali karena MODEL_ORDER disalin dari generator
v1 tanpa dicek ulang. Setiap pembangun figur revisi WAJIB mengimpor dari sini
dan tidak menuliskan daftar arm sendiri.
"""

# Arm utama, semuanya ImageNet-1k. Urutan ini dipakai di semua figur dan legenda.
MAIN_ARMS = ["resnet50", "convnext_tiny_in1k", "vit_small_in1k", "swin_tiny"]

# Checkpoint yang sebenarnya dimuat tiap arm, untuk dicocokkan dengan log run.
PRETRAIN_TAG = {
    "resnet50": "a1_in1k",
    "convnext_tiny_in1k": "fb_in1k",
    "vit_small_in1k": "augreg_in1k",
    "swin_tiny": "ms_in1k",
}

MLAB = {
    "resnet50": "ResNet-50",
    "convnext_tiny_in1k": "ConvNeXt-T",
    "vit_small_in1k": "ViT-S",
    "swin_tiny": "Swin-T",
}

# Warna dikunci di sini. analyze_extra.py memakai #8172B3 untuk Swin sedangkan
# figur revisi memakai #8172B2; dipilih #8172B2 supaya seluruh figur seragam.
MCOLOR = {
    "resnet50": "#4C72B0",
    "convnext_tiny_in1k": "#55A868",
    "vit_small_in1k": "#C44E52",
    "swin_tiny": "#8172B2",
}

DSLAB = {"flowers102": "Flowers-102", "cub200": "CUB-200"}

# Pasangan grayscale-training (R2.3). Dipakai arm setara supaya clean dan gray
# berangkat dari pretrain yang sama, sehingga selisihnya mengukur warna dan
# bukan mengukur data pretraining.
GRAY_ARM = {
    "resnet50": "resnet50_gray",
    "convnext_tiny_in1k": "convnext_tiny_in1k_gray",
    "vit_small_in1k": "vit_small_in1k_gray",
    "swin_tiny": "swin_tiny_gray",
}

# Checkpoint NON-setara. Hanya boleh muncul di tabel confound A6, tidak pernah
# di tabel atau figur utama.
CONFOUND_ARMS = {
    "convnext_tiny": "in12k_ft_in1k",
    "vit_small": "augreg_in21k_ft_in1k",
}


def check(df, col="model"):
    """Berhenti kalau sebuah frame membawa arm non-setara ke jalur figur utama."""
    bad = sorted(set(df[col].unique()) & set(CONFOUND_ARMS))
    if bad:
        raise SystemExit(
            f"arm non-setara masuk jalur figur utama: {bad}. "
            f"Pakai {MAIN_ARMS} (lihat revision/arms.py, komentar R1.2)."
        )
