"""V6 (R2.1): heatmap RGB vs grayscale per arsitektur.
  V6a ResNet  -> Grad-CAM di layer4
  V6b ConvNeXt-> Grad-CAM di stage terakhir
  V6c ViT     -> attention rollout
  V6d Swin    -> Grad-CAM di stage terakhir

Sampel: beberapa gambar per bin colorfulness (low/mid/high). Output grid PNG:
  figures/V6_gradcam_<ds>.png  (baris=sampel, kolom = RGB | RGB-CAM | gray | gray-CAM per model)

CATATAN: struktur modul timm beda per-model -> target layer di-resolve dgn fallback +
try/except. WAJIB smoke-test di checkpoint FINAL (setelah consist). Seed0 saja.
  python revision/eval_gradcam.py --n_per_bin 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, chdir_to_root
from src import data as data_mod
from src import models as models_mod

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
ARMS = [("resnet50", "ResNet-50", "gradcam"),
        ("convnext_tiny_in1k", "ConvNeXt-T", "gradcam"),
        ("vit_small_in1k", "ViT-S", "rollout"),
        ("swin_tiny", "Swin-T", "gradcam")]


SWIN_TARGET = "final"   # "final" (stage akhir 7x7) | "penult" (stage ke-2 akhir 14x14)
CAM_METHOD = "gradcam"  # "gradcam" | "layercam" (element-wise grad, lebih tajam)
# Normalisasi tampilan: "minmax" (mentah) | "pct" (persentil, tekan lantai difus).
# "pct" diterapkan SERAGAM ke keempat arsitektur -> konvensi tampilan, bukan
# perlakuan khusus buat satu model.
NORM_MODE = "pct"
PCT_LO, PCT_HI = 50, 99


def _norm(cam):
    if NORM_MODE == "pct":
        lo = np.percentile(cam, PCT_LO)
        hi = np.percentile(cam, PCT_HI)
        return np.clip((cam - lo) / (hi - lo + 1e-8), 0, 1)
    return (cam - cam.min()) / (np.ptp(cam) + 1e-8)


def correct_by_all(rdir, ds, seed=0):
    """image_id yang diklasifikasi BENAR (RGB clean) oleh KEEMPAT model di seed ini.
    Heatmap baru bermakna kalau modelnya memang mengenali objeknya."""
    arms = ["resnet50", "convnext_tiny_in1k", "vit_small_in1k", "swin_tiny"]
    ok, truth = None, None
    for a in arms:
        f = rdir / "preds" / f"{ds}_{a}_seed{seed}.csv"
        d = pd.read_csv(f).set_index("image_id")
        good = set(d.index[d.y_true == d.pred_clean])
        ok = good if ok is None else (ok & good)
        truth = d["y_true"] if truth is None else truth
    return ok, truth


def select_samples(rdir, tdir, ds, mode, n=6, seed=0, ref_arm="vit_small_in1k"):
    """Return (picks, meta_df). picks = list image_id. meta = keterangan caption."""
    ok, truth = correct_by_all(rdir, ds, seed)
    v7 = pd.read_csv(tdir / f"V7_perspecies_CR_{ds}.csv")
    v7 = v7[v7.arm == ref_arm].set_index("y_true")["CR_mean"].sort_values()
    cls_of = truth  # image_id -> y_true

    def pick_img(c, k=1):
        cand = sorted(i for i in ok if cls_of.get(i, -1) == c)
        return cand[:k]

    rows, picks = [], []
    if mode == "stratified":
        lo = list(v7.index[:2])
        mid_i = len(v7) // 2
        mid = list(v7.index[mid_i - 1:mid_i + 1])
        hi = list(v7.index[-2:])
        for band, cls in [("high", hi), ("median", mid), ("low", lo)]:
            for c in cls:
                im = pick_img(c)
                if im:
                    picks.append(im[0])
                    rows.append(dict(image_id=im[0], species_class=int(c),
                                     CR_band=band, CR_ref=float(v7[c])))
    elif mode == "posrange":
        # Varian pembanding: 6 kelas disebar merata dari MEDIAN ke MAKSIMUM CR
        # (ekor negatif sengaja tidak disertakan). Caption WAJIB menyebut
        # "median-to-maximum range", JANGAN "spanning the CR distribution".
        n_all = len(v7)
        qs = np.linspace(0.50, 1.0, n)
        idx = sorted({min(n_all - 1, int(round(q * (n_all - 1)))) for q in qs})
        for i in idx:
            c = int(v7.index[i])
            im = pick_img(c)
            if im:
                picks.append(im[0])
                pct = 100.0 * i / (n_all - 1)
                rows.append(dict(image_id=im[0], species_class=c,
                                 CR_band=f"pct{pct:.0f}", CR_ref=float(v7.iloc[i])))
    elif mode == "seeded":
        rng = np.random.default_rng(seed)
        cls = sorted(rng.choice(np.array(v7.index), size=n, replace=False).tolist())
        for c in cls:
            im = pick_img(c)
            if im:
                picks.append(im[0])
                rows.append(dict(image_id=im[0], species_class=int(c),
                                 CR_band="random", CR_ref=float(v7[c])))
    elif mode == "single":
        c = int(v7.index[len(v7) // 2])          # kelas CR-median
        for im in pick_img(c, n):
            picks.append(im)
            rows.append(dict(image_id=im, species_class=c, CR_band="median (single species)",
                             CR_ref=float(v7[c])))
    else:
        raise SystemExit(f"mode tak dikenal: {mode}")
    return picks, pd.DataFrame(rows)


def target_layer(model, arm):
    """Ambil modul target Grad-CAM (feature map terakhir) dengan fallback.
    Swin: stage akhir 7x7 == window_size 7 -> satu window = seluruh peta, jadi
    efektif GLOBAL (atribusi difus). Opsi 'penult' pakai stage 14x14 yang
    window-nya benar-benar lokal -> peta lebih tajam."""
    if "swin" in arm and hasattr(model, "layers"):
        return model.layers[-1] if SWIN_TARGET == "final" else model.layers[-2]
    for attr in ("layer4", "stages", "layers", "norm_pre"):
        if hasattr(model, attr):
            m = getattr(model, attr)
            try:
                return m[-1]
            except (TypeError, KeyError):
                return m
    # fallback: modul konv/norm terakhir
    last = None
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.LayerNorm)):
            last = mod
    return last


def gradcam(model, x, layer):
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.zero_grad()
    out = model(x)
    cls = out.argmax(1)
    out[0, cls[0]].backward()
    h1.remove(); h2.remove()
    a, g = acts["a"], grads["g"]
    if a.dim() == 4:                       # conv-like
        # timm Swin keluar channels-last (B,H,W,C); ConvNeXt/ResNet (B,C,H,W).
        # Deteksi: kalau dim1==dim2 dan dim3 lebih besar -> channels-last, permute.
        if a.shape[1] == a.shape[2] and a.shape[3] > a.shape[1]:
            a = a.permute(0, 3, 1, 2).contiguous()
            g = g.permute(0, 3, 1, 2).contiguous()
        if CAM_METHOD == "layercam":
            # LayerCAM (Jiang et al., IEEE TIP 2021): bobot gradien ELEMENT-WISE,
            # bukan global-average-pooled -> detail spasial tidak luruh di layer
            # yang sudah ter-mix global (Swin stage akhir).
            cam = F.relu((F.relu(g) * a).sum(1, keepdim=True))
        else:
            w = g.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * a).sum(1, keepdim=True))
    else:                                  # B,N,C (token) -> buang cls token, reshape
        t = a[:, 1:, :]; gg = g[:, 1:, :]
        w = gg.mean(1, keepdim=True)
        c = F.relu((w * t).sum(-1))
        s = int(round(c.shape[1] ** 0.5))
        cam = c[:, :s * s].reshape(1, 1, s, s)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    return _norm(cam)


def attn_rollout(model, x):
    """Attention rollout ViT: kumpulkan attn softmax tiap blok, kalikan."""
    attns = []
    hooks = []
    saved_fused = []
    for blk in getattr(model, "blocks", []):
        attn = getattr(blk, "attn", None)
        if attn is not None and hasattr(attn, "attn_drop"):
            # timm default fused_attn=True (F.sdpa) -> attn_drop tak pernah lihat
            # matriks attention. Matikan sementara supaya rollout dapat bobotnya.
            saved_fused.append((attn, getattr(attn, "fused_attn", None)))
            if hasattr(attn, "fused_attn"):
                attn.fused_attn = False
            hooks.append(attn.attn_drop.register_forward_hook(
                lambda m, i, o: attns.append(i[0].detach())))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    for attn, fv in saved_fused:          # kembalikan setting semula
        if fv is not None:
            attn.fused_attn = fv
    if not attns:
        return np.zeros(x.shape[-2:])
    result = torch.eye(attns[0].shape[-1], device=attns[0].device)
    for a in attns:
        a = a.max(1)[0]                    # max antar-head (lebih tajam drpd mean)
        # discard-ratio (praktik baku rollout): buang atensi terlemah supaya
        # lantai-noise latar tidak menumpuk sepanjang perkalian antar-layer.
        k = max(1, int(a.shape[-1] * 0.10))
        thr = a.topk(k, dim=-1).values[..., -1:]
        a = torch.where(a >= thr, a, torch.zeros_like(a))
        a = a + torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        result = a[0] @ result
    mask = result[0, 1:]                   # cls -> patch
    s = int(round(mask.shape[0] ** 0.5))
    # Upsample bilinear SAMA seperti jalur Grad-CAM (yang juga membesarkan peta
    # kasar 7x7). Sebelumnya pakai PIL resize -> tepi keras/kotak-kotak, jadi
    # perlakuan tampilannya beda sendiri dari tiga arsitektur lain.
    mask = mask[:s * s].reshape(1, 1, s, s)
    mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear",
                         align_corners=False)[0, 0].cpu().numpy()
    return _norm(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cub200", "flowers102"])
    ap.add_argument("--n_per_bin", type=int, default=2)
    ap.add_argument("--select", choices=["stratified","seeded","single","posrange"], default="stratified")
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--sel_seed", type=int, default=0)
    args = ap.parse_args()
    chdir_to_root()
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import pandas as pd
    rdir = Path(cfg["paths"]["results"])
    fdir = Path(cfg["paths"]["figures"]); fdir.mkdir(parents=True, exist_ok=True)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    tdir = Path(cfg["paths"]["tables"])
    for ds in args.datasets:
        picks, meta = select_samples(rdir, tdir, ds, args.select, n=args.n_samples,
                                     seed=args.sel_seed)
        meta.to_csv(fdir / f"V6_samples_{args.select}_{ds}.csv", index=False)
        print(f"  [select={args.select}] {len(picks)} sampel -> "
              f"{fdir}/V6_samples_{args.select}_{ds}.csv")
        for _, r in meta.iterrows():
            print(f"    class {r.species_class:>4}  band={r.CR_band:<24} "
                  f"CR_ref={r.CR_ref:+.4f}  {r.image_id}")
        crmap = {r.image_id: float(r.CR_ref) for _, r in meta.iterrows()}
        clsmap = {r.image_id: int(r.species_class) for _, r in meta.iterrows()}
        dset, ids = data_mod.get_eval_dataset_with_ids(ds, cfg, normalize=False)
        id2idx = {i: k for k, i in enumerate(ids)}
        try:
            models_loaded = {arm: models_mod.load_checkpoint(
                Path(cfg["paths"]["checkpoints"]) / f"{ds}_{arm}_seed0.pt",
                arm, cfg, cfg["datasets"][ds]["num_classes"], device)[0]
                for arm, _, _ in ARMS}
        except FileNotFoundError as e:
            print(f"  {ds}: checkpoint belum lengkap -> {e}"); continue

        nrow, ncol = len(picks), 1 + 2 * len(ARMS)
        # \textwidth ieeeaccess.cls = 6.989 in (SAMA dgn fig1-5). Ukuran figur = ukuran
        # cetak supaya font terlihat = font tercetak (tanpa scaling). Font >= 8 pt.
        W = 6.989
        panel = W / ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(W, panel * nrow + 0.35),
                                 squeeze=False)
        for r, iid in enumerate(picks):
            if iid not in id2idx:
                continue
            x0, _ = dset[id2idx[iid]]
            x0 = x0.unsqueeze(0).to(device)
            gray = x0.mean(1, keepdim=True).repeat(1, 3, 1, 1)
            img = x0[0].permute(1, 2, 0).cpu().numpy()
            axes[r][0].imshow(img)
            # CUB menaruh nama spesies di path ("022.Chuck_wills_widow/..."), jadi
            # baris bisa diberi nama. Flowers-102 memakai nama berkas datar
            # ("image_06734.jpg") dan datasetnya TIDAK mengirim daftar nama kelas
            # resmi, jadi baris dilabeli indeks kelas. Daftar nama umum yang
            # beredar tidak dipakai: urutannya terhadap indeks label tidak terjamin
            # dan salah melabeli spesies di figur terbit tidak bisa ditarik.
            # nama pendek SATU baris (Ers) + font kecil supaya muat di kolom ~0.78 in.
            SHORT = {"Magnolia Warbler": "Magnolia", "Green Violetear": "Green Violet.",
                     "Red faced Cormorant": "Red Cormo.", "Chuck will Widow": "Chuck Widow",
                     "Whip poor Will": "Whip Will", "Mockingbird": "Mockingbird"}
            if "/" in iid:
                _full = iid.split("/")[0].split(".", 1)[-1].replace("_", " ")
                _sp = SHORT.get(_full, _full.split()[0])
            else:
                _sp = "class %d" % clsmap.get(iid, -1)
            _cr = crmap.get(iid, float("nan"))
            axes[r][0].set_title("%s\nCR = %+.2f" % (_sp, _cr),
                                 fontsize=7, linespacing=1.3)
            for c, (arm, lab, method) in enumerate(ARMS):
                model = models_loaded[arm]
                for j, xin in enumerate([x0, gray]):
                    xn = (xin - mean) / std
                    try:
                        if method == "rollout":
                            cam = attn_rollout(model, xn)
                        else:
                            cam = gradcam(model, xn.clone().requires_grad_(True),
                                          target_layer(model, arm))
                    except Exception as ex:
                        cam = np.zeros(x0.shape[-2:]); print(f"    {arm} fail: {ex}")
                    ax = axes[r][1 + 2 * c + j]
                    ax.imshow(xin[0].permute(1, 2, 0).cpu().numpy())
                    ax.imshow(cam, cmap="jet", alpha=0.5)
                    if r == 0:
                        # label pendek supaya muat 8 pt di kolom ~0.78 in (9 kolom /
                        # 6.989 in). Nama penuh ada di caption.
                        _short = {"ResNet-50": "ResNet", "ConvNeXt-T": "ConvNeXt"}.get(lab, lab)
                        ax.set_title(f"{_short}\n{'RGB' if j == 0 else 'gray'}", fontsize=8)
        for ax in axes.ravel():
            ax.axis("off")
        fig.tight_layout()
        p = fdir / f"V6_gradcam_{args.select}_{ds}.png"
        # TANPA bbox_inches="tight": jaga lebar keluaran = 6.989 in (8 pt = 8 pt cetak),
        # sama seperti build_fig1-5.
        fig.savefig(p, dpi=600); plt.close(fig)
        print(f"  -> {p}")
        try:
            from PIL import Image as _Im
            _im = _Im.open(p); _w = _im.size[0] / _im.info["dpi"][0]
            print(f"     {_im.size[0]}x{_im.size[1]} px @ {round(_im.info['dpi'][0])} dpi = {_w:.3f} in")
        except Exception:
            pass


if __name__ == "__main__":
    main()
