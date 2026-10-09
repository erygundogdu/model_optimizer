"""Prune a YOLOv7 checkpoint using user-supplied paths."""
import argparse
from pathlib import Path

from optimizer.paths import DEFAULT_YOLOV7_REPO, resolve_yolov7_repo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--repo-path', type=Path, default=DEFAULT_YOLOV7_REPO,
                        help='YOLOv7 source directory (default: sibling ../yolov7)')
    parser.add_argument('--weights', type=Path, required=True, help='input YOLOv7 checkpoint')
    parser.add_argument('--output', type=Path, default=Path('pruned.pt'), help='output checkpoint')
    parser.add_argument('--img-size', type=int, nargs='+', default=[1280],
                        help='one square size or HEIGHT WIDTH (default: 1280)')
    parser.add_argument('--pruning-ratio', type=float, default=0.1)
    parser.add_argument('--device', default=None, help='cpu, cuda, or cuda:0 (default: auto)')
    args = parser.parse_args(argv)
    if len(args.img_size) not in (1, 2) or any(size <= 0 for size in args.img_size):
        parser.error('--img-size requires one or two positive integers')
    if not 0 < args.pruning_ratio < 1:
        parser.error('--pruning-ratio must be between 0 and 1')
    try:
        repo = resolve_yolov7_repo(args.repo_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        parser.error(f'Checkpoint not found: {weights}')
    output = args.output.expanduser().resolve()
    if output == weights:
        parser.error('--output must differ from --weights')

    import torch
    from optimizer.adapters.yolov7 import YOLOv7Adapter
    from optimizer.pipeline import run_gamma_pruning_pipeline

    height, width = args.img_size * 2 if len(args.img_size) == 1 else args.img_size
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    run_gamma_pruning_pipeline(
        adapter=YOLOv7Adapter(repo_path=repo), weights_path=str(weights),
        output_path=str(output), input_shape=(1, 3, height, width),
        pruning_ratio=args.pruning_ratio, device=device,
    )


if __name__ == '__main__':
    main()
