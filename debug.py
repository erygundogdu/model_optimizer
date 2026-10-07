import sys
import torch
import torch.nn as nn

sys.path.insert(0, r"C:\Users\statb\Desktop\yolov7")

pth = "pruned.pt"
ckpt = torch.load(pth, map_location="cpu", weights_only=False)

model = ckpt["model"]

detect = model.model[-1]
#print(list(detect.modules()))

for idx, layer in enumerate(model.model):
    for name, m in layer.named_modules():
        if isinstance(m, nn.Conv2d):
            path = f"model.{idx}" + (f".{name}" if name else "")
            print(f"Layer {idx}: {path} — {tuple(m.weight.shape)}")
