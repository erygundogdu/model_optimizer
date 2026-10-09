"""Print convolution shapes from a YOLOv7 checkpoint."""
import argparse
from pathlib import Path
import sys

from optimizer.paths import DEFAULT_YOLOV7_REPO, resolve_yolov7_repo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--repo-path', type=Path, default=DEFAULT_YOLOV7_REPO)
    parser.add_argument('--weights', type=Path, default=Path('pruned.pt'))
    args = parser.parse_args(argv)
    try:
        repo = resolve_yolov7_repo(args.repo_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        parser.error(f'Checkpoint not found: {weights}')
    sys.path.insert(0, str(repo))

    import torch

    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    model = checkpoint.get('ema') if isinstance(checkpoint, dict) else checkpoint
    if model is None:
        model = checkpoint.get('model')
    if not isinstance(model, torch.nn.Module) or not hasattr(model, 'model'):
        parser.error('Expected a checkpoint containing a YOLOv7 model module')
    for idx, layer in enumerate(model.model):
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                path = f'model.{idx}' + (f'.{name}' if name else '')
                print(f'Layer {idx}: {path} - {tuple(module.weight.shape)}')


if __name__ == '__main__':
    main()
