import torch

from optimizer.adapters.yolov7 import (
    YOLOv7Adapter
)

from optimizer.pipeline import (
    run_gamma_pruning_pipeline
)


YOLOV7_REPO = (
    r"C:\Users\eray.gundogdu\Desktop\yolov7"
)

WEIGHTS = (
    r"C:\Users\eray.gundogdu\Desktop\model_optimizer\best.pt"
)

OUTPUT = (
    r"C:\Users\eray.gundogdu\Desktop"
    r"\model_optimizer\pruned.pt"
)
#def channel_prune(model, equal, image.T)
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


INPUT_SHAPE = (
    1,
    3,
    1664,
    1664
)     
PRUNING_RATIO = 0.1

def main():

    adapter = YOLOv7Adapter(repo_path=YOLOV7_REPO)

    run_gamma_pruning_pipeline(adapter=adapter,weights_path=WEIGHTS,output_path=OUTPUT,input_shape=INPUT_SHAPE,pruning_ratio=PRUNING_RATIO,
                               device=DEVICE,
)


if __name__ == "__main__":
    main()