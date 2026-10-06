import torch
import torch.nn as nn

pth = "pruned.pt"
ckpt = torch.load(pth,weights_only=False)

model = ckpt["model"]

detect = model.model[-1]
#print(list(detect.modules()))

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        print(m.weight.shape)