# Model optimizer

Prune YOLOv7 channels using BatchNorm gamma scores, then fine-tune the saved
architecture. All input and output paths are supplied on the command line.
No source edits are needed after cloning. Environment setup is outside this guide.

## Prerequisites

- Python 3.10 or newer and a working PyTorch/torchvision installation compatible
  with your selected YOLOv7 checkout. CUDA is optional; CPU runs are supported.
- A **separate YOLOv7 source checkout**. By default it must be named `yolov7` and
  sit beside this repository, not inside it:

  ```text
  workspace/
    model_optimizer/
      optimize.py
      finetune_pruned.py
    yolov7/
      models/
      utils/
      test.py
      requirements.txt
  ```

  For standard YOLOv7 models, obtain the source with
  `git clone https://github.com/WongKinYiu/yolov7.git ../yolov7` from this folder.
  For a checkpoint trained with custom layers (for example SPD), use the same
  compatible fork and layer definitions used to train it. A stock checkout cannot
  deserialize classes that only exist in your custom fork. Model classes,
  dataloaders, losses, validation and logging come from this external checkout;
  it does not need its own `finetune_pruned.py` or a custom `freeze_bn` helper.
- That checkout's dependencies and the optimizer's extra dependency:

  ```sh
  python -m pip install -r ../yolov7/requirements.txt
  python -m pip install -r requirements.txt
  ```

  Substitute your checkout's path in the first command when using `--repo-path`.
  Skip the first command if its dependencies are already installed. The optimizer
  uses Torch-Pruning 1.6.1. Older stock YOLOv7 utilities use NumPy aliases removed
  in NumPy 1.24 and full-model `torch.load` defaults changed in PyTorch 2.6.
  Such checkouts need NumPy below 1.24 and PyTorch below 2.6 (with matching
  torchvision), or a checkout updated for those APIs. This repository does not
  modify the external checkout. Python 3.10 is the practical choice for those
  older dependency versions.
- Your own YOLOv7 `.pt` checkpoint containing a model module, rather than only a
  state dictionary. Fine-tuning also requires a dataset YAML and YOLO-format
  images/labels. Its class count must match the checkpoint's detection head.
  Checkpoints and datasets are not included in Git.

`--repo-path /path/to/yolov7` overrides the sibling default in all three tools.
Quote paths containing spaces. The sibling default is located relative to this
repository, regardless of where you launch the command.

## Prune a checkpoint

Run from this repository:

```sh
python optimize.py --weights /path/to/best.pt --output pruned.pt --img-size 640 --pruning-ratio 0.1 --device cpu
```

For a different YOLOv7 checkout or GPU:

```sh
python optimize.py --repo-path /path/to/yolov7 --weights /path/to/best.pt --output /path/to/results/pruned.pt --img-size 1280 --pruning-ratio 0.1 --device cuda:0
```

`--weights` is required. Defaults are output `pruned.pt` in your current directory,
input size 1280, pruning ratio 0.1, and CUDA when available, otherwise CPU.
`--img-size HEIGHT WIDTH` accepts rectangular inputs. Use dimensions supported
by your model, normally multiples of its largest stride. Output directories are
created automatically. The output path must differ from the input checkpoint.
The pipeline checks forward passes before and after pruning and saves pruning
metadata. Safety constraints can reduce the achieved pruning ratio.

## Fine-tune the pruned checkpoint

```sh
python finetune_pruned.py --weights pruned.pt --data /path/to/data.yaml --img-size 640 640 --batch-size 8 --epochs 30 --device 0 --name pruned_finetune
```

Use `--device cpu` for CPU training. Fine-tuning loads the saved model directly;
rebuilding it from the original architecture YAML would lose the reduced widths.
`--cfg` is unsupported. A new run resets optimizer and EMA state.

File arguments (`--weights`, `--data`, `--hyp`, `--project`, `--resume`) resolve
from your current directory. Relative train/val entries **inside the dataset
YAML resolve from the YOLOv7 checkout**, following its dataloader behavior. Use
absolute image-directory or image-list paths in your dataset YAML for portability.
For example (replace paths and class names):

```yaml
train: /path/to/dataset/images/train
val: /path/to/dataset/images/val
nc: 2
names: [class_a, class_b]
```

Default weights are `pruned.pt` in this repository. Default hyperparameters are
`configs/hyp.finetune.yaml`, with 30 epochs, batch size 8, image size 640, initial
learning rate `1e-4`, and one warmup epoch. `--lr0`, `--warmup-epochs`, and
`--warmup-bias-lr` override the YAML for a new run. BN parameters and statistics
train by default; `--freeze_bn` freezes both.

Results are written under this repository's
`runs/finetune/pruned_finetune/` (incremented if it already exists): best/last
checkpoints, results, plots, TensorBoard logs, `hyp.yaml`, and `opt.yaml`.
Override the destination with `--project` and `--name`.

```sh
python finetune_pruned.py --resume runs/finetune/pruned_finetune/weights/last.pt
python finetune_pruned.py --resume
```

Bare `--resume` searches the local fine-tuning project or the given `--project`.
Resume an interrupted run: YOLOv7 strips optimizer state at successful completion.
Use `--weights` to start further fine-tuning from a completed checkpoint.
The launcher uses the active Python interpreter for its child process and
configures YOLOv7 imports for Windows dataloader workers as well.

## Inspect a checkpoint and view options

```sh
python debug.py --weights pruned.pt --repo-path /path/to/yolov7
python optimize.py --help
python finetune_pruned.py --help
python debug.py --help
```

Inspection prints convolution weight shapes. All help commands work before
installing training dependencies or obtaining the YOLOv7 checkout. Missing
source directories and checkpoints produce errors describing the required path.
