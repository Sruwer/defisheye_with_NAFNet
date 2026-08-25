from __future__ import annotations
import importlib
from pathlib import Path
import numpy as np
from .base import ImageRestorer

class NAFNetRestorer(ImageRestorer):
    def __init__(self, cfg: dict):
        import torch
        self.torch = torch
        requested = cfg.get("device", "cuda")
        self.device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
        self.fp16 = bool(cfg.get("fp16", True) and self.device.type == "cuda")
        module = importlib.import_module(cfg["architecture_module"])
        cls = getattr(module, cfg.get("architecture_class", "NAFNet"))
        self.model = cls(img_channel=3, width=cfg["width"], enc_blk_nums=cfg["enc_blk_nums"],
                         middle_blk_num=cfg["middle_blk_num"], dec_blk_nums=cfg["dec_blk_nums"])
        path = Path(cfg["weights"])
        if not path.is_file(): raise FileNotFoundError(f"NAFNet weights not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("params", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def restore(self, image: np.ndarray) -> np.ndarray:
        torch = self.torch
        rgb = image[..., ::-1].copy()
        x = torch.from_numpy(rgb).permute(2,0,1).float().div_(255).unsqueeze(0).to(self.device)
        h, w = x.shape[-2:]; ph, pw = (-h)%16, (-w)%16
        x = torch.nn.functional.pad(x, (0,pw,0,ph), mode="reflect")
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            y = self.model(x).clamp_(0,1)[..., :h, :w]
        return (y[0].permute(1,2,0).float().cpu().numpy()[..., ::-1]*255).round().astype(np.uint8)
