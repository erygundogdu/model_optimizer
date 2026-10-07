# Model optimizer

`optimize.py` performs BN gamma channel pruning and saves the reduced model to
`pruned.pt`. Fine-tuning loads that saved model directly; rebuilding it from the
original architecture YAML would lose the reduced channel widths.

## Fine-tune a pruned checkpoint

Activate the same TRAIN environment used for YOLOv7, then run from this folder:

```powershell
python finetune_pruned.py --weights pruned.pt --data ../yolov7/data/SINOP_1664/data.yaml --img-size 1664 1664 --batch-size 8 --epochs 30 --device 0 --name pruned_finetune
```

Replace the example dataset YAML and image sizes with those used for your model.
The dataset class count must match the checkpoint's detection head.

The default YOLOv7 repository is the sibling `../yolov7` directory. To use another:

```powershell
python finetune_pruned.py --repo-path C:/path/to/yolov7 --data C:/path/to/data.yaml --device 0
```

The local training loop lives in `optimizer/finetuning/yolov7_train.py`, copied
from the existing YOLOv7 `finetune_pruned.py`. It uses that repository's model
classes, dataloaders, losses, validation, EMA, and logging utilities. Keep the
compatible YOLOv7 source and its training dependencies installed; this project
does not vendor those utilities or call the external fine-tuning script.

The launcher runs training in a child process using the active Python environment.
File arguments (`--weights`, `--data`, `--hyp`, `--project`, `--resume`) resolve
from the directory where you launch the command. Relative train/val entries
inside dataset YAMLs resolve from the YOLOv7 repository, matching the old script.
Use absolute dataset entries if you want to avoid that dependency.

Defaults: `pruned.pt` in this repository, local `configs/hyp.finetune.yaml`,
30 epochs, batch size 8, image size 640, learning rate `1e-4`, and a one-epoch
warmup. The hyperparameter YAML was copied from your YOLOv7 p5 configuration,
including its augmentation and loss settings. The command-line fine-tuning
learning rate and warmup values override the YAML for a new run.

BN parameters and statistics train by default. `--freeze_bn` freezes BN behavior
using the existing YOLOv7 helper. `--cfg` is unsupported: architecture comes from
the checkpoint. A new run resets the optimizer and EMA; use `--resume` only for
an interrupted fine-tuning run.

Results are saved under `runs/finetune/pruned_finetune/` (incremented if it exists):

- `weights/best.pt` and `weights/last.pt`
- `results.txt`, plots, TensorBoard logs, `hyp.yaml`, and `opt.yaml`

```powershell
python finetune_pruned.py --resume runs/finetune/pruned_finetune/weights/last.pt
python finetune_pruned.py --resume
python finetune_pruned.py --help
```

Bare `--resume` searches the local fine-tuning project, or the given `--project`.
The original trainer strips optimizer state from best/last at successful
completion, so resume an interrupted run before that finalization. Start a new
run with `--weights` to fine-tune a completed checkpoint further.
