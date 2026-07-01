"""Model zoo: semua pretrained ImageNet-1k via timm, param ~22-28M (comparable).

| key            | timm_name                      | role                         |
| resnet50       | resnet50                       | CNN klasik                   |
| convnext_tiny  | convnext_tiny                  | CNN modern (resep ViT) KONTROL|
| vit_small      | vit_small_patch16_224          | ViT murni                    |
| swin_tiny      | swin_tiny_patch4_window7_224   | atensi hierarkis             |
"""
import timm
import torch


def build_model(model_key, cfg, num_classes, pretrained=True):
    timm_name = cfg["models"][model_key]["timm_name"]
    model = timm.create_model(timm_name, pretrained=pretrained,
                              num_classes=num_classes)
    return model


def load_checkpoint(ckpt_path, model_key, cfg, num_classes, device="cuda"):
    """Bangun model + load bobot dari checkpoint (tanpa pretrained download)."""
    model = build_model(model_key, cfg, num_classes, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    return model, ckpt


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_flops(model, image_size=224, device="cpu"):
    """GFLOPs untuk 1 forward pass (batch=1). Pakai thop kalau ada."""
    try:
        from thop import profile
        model = model.to(device).eval()
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs * 2 / 1e9  # MACs -> FLOPs -> GFLOPs
    except Exception as e:
        print(f"[measure_flops] gagal ({e}), return None")
        return None
