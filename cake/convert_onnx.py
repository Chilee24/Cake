# export_onnx.py
import torch
import torch.nn as nn
from pathlib import Path

from cake import BioX3D_Student
from rnn import MROAD

# ===============================
# ONNX WRAPPERS
# ===============================

class BioX3D_ONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        _, _, _, _, rgb_emb, flow_emb = self.model(x, return_embeddings=True)
        return rgb_emb, flow_emb


class MiniROAD_ONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, rgb, flow):
        out = self.model(rgb, flow)
        return out["logits"]


# ===============================
# EXPORT
# ===============================

def main():
    out_dir = Path("onnx_models")
    out_dir.mkdir(exist_ok=True)

    # -------- BioX3D --------
    biox3d = BioX3D_Student()

    biox3d_onnx = BioX3D_ONNX(biox3d)

    dummy_clip = torch.randn(1, 3, 13, 224, 224)

    torch.onnx.export(
        biox3d_onnx,
        dummy_clip,
        out_dir / "biox3d.onnx",
        opset_version=14,
        input_names=["video"],
        output_names=["rgb_embedding", "flow_embedding"]
    )

    print("✅ Exported BioX3D")

    # -------- MiniROAD --------
    cfg = {
        'no_rgb': False,
        'no_flow': False,
        'rgb_type': 'rgb_kinetics_x3d',
        'flow_type': 'flow_kinetics_x3d',
        'hidden_dim': 512,
        'num_layers': 2,
        'num_classes': 10,
        'window_size': 30,
        'embedding_dim': 256,
        'dropout': 0.0,
    }

    miniroad = MROAD(cfg)
    miniroad_onnx = MiniROAD_ONNX(miniroad)

    dummy_rgb = torch.randn(1, 128, 2048)
    dummy_flow = torch.randn(1, 128, 2048)

    torch.onnx.export(
        miniroad_onnx,
        (dummy_rgb, dummy_flow),
        out_dir / "miniroad.onnx",
        opset_version=14,
        input_names=["rgb_features", "flow_features"],
        output_names=["logits"],
    )

    print("✅ Exported MiniROAD")


if __name__ == "__main__":
    main()
