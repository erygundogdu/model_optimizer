"""Launch the local fine-tuner with the external YOLOv7 utilities."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from optimizer.paths import PROJECT_DIR, DEFAULT_YOLOV7_REPO, resolve_yolov7_repo


def build_command(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--repo-path", type=Path, default=DEFAULT_YOLOV7_REPO,
        help="YOLOv7 repository providing models, datasets, losses and validation",
    )
    options, training_args = parser.parse_known_args(argv)
    if '--help' in training_args or '-h' in training_args:
        from optimizer.finetuning.cli import build_parser
        help_parser = build_parser()
        help_parser.add_argument('--repo-path', type=Path, default=DEFAULT_YOLOV7_REPO,
                                 help='YOLOv7 source directory (default: sibling ../yolov7)')
        help_parser.parse_args(argv)
    try:
        repo = resolve_yolov7_repo(options.repo_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    # Resolve user paths before the child switches to the YOLOv7 directory.
    # Paths inside dataset YAMLs keep the original YOLOv7 working directory.
    path_flags = {"--weights", "--data", "--hyp", "--project", "--resume", "--cfg"}
    resolved_args = []
    provided = set()
    index = 0
    while index < len(training_args):
        argument = training_args[index]
        flag, separator, value = argument.partition("=")
        provided.add(flag)
        if flag in path_flags:
            if separator:
                if value:
                    value = str(Path(value).expanduser().resolve())
                resolved_args.append(f"{flag}={value}")
            elif index + 1 < len(training_args) and not training_args[index + 1].startswith("-"):
                value = training_args[index + 1]
                resolved_args.extend([flag, str(Path(value).expanduser().resolve()) if value else value])
                index += 1
            else:
                resolved_args.append(argument)
        else:
            resolved_args.append(argument)
        index += 1

    defaults = {
        "--weights": PROJECT_DIR / "pruned.pt",
        "--hyp": PROJECT_DIR / "configs" / "hyp.finetune.yaml",
        "--project": PROJECT_DIR / "runs" / "finetune",
    }
    for flag, value in defaults.items():
        if flag not in provided:
            resolved_args.extend([flag, str(value)])

    from optimizer.finetuning.cli import build_parser
    training_parser = build_parser()
    training = training_parser.parse_args(resolved_args)
    if training.cfg:
        training_parser.error('--cfg is unsupported: architecture comes from --weights')
    if not training.resume:
        if not training.data:
            training_parser.error('--data is required for a new fine-tuning run')
        for flag in ('weights', 'data', 'hyp'):
            path = Path(getattr(training, flag))
            if not path.is_file():
                training_parser.error(f'--{flag} file not found: {path}')
    elif isinstance(training.resume, str) and not Path(training.resume).is_file():
        training_parser.error(f'--resume checkpoint not found: {training.resume}')

    backend = PROJECT_DIR / "optimizer" / "finetuning" / "yolov7_train.py"
    command = [sys.executable, str(backend), *resolved_args]
    env = os.environ.copy()
    # The child and Windows dataloader workers must resolve models.* identically.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(PROJECT_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return command, repo, env


def main(argv=None):
    command, repo, env = build_command(sys.argv[1:] if argv is None else argv)
    return subprocess.call(command, cwd=repo, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
