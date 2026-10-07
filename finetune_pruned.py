"""Launch the local fine-tuner with the external YOLOv7 utilities."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent


def build_command(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--repo-path", type=Path, default=PROJECT_DIR.parent / "yolov7",
        help="YOLOv7 repository providing models, datasets, losses and validation",
    )
    options, training_args = parser.parse_known_args(argv)
    repo = options.repo_path.resolve()
    for required in ("models/yolo.py", "utils/datasets.py", "test.py"):
        if not (repo / required).is_file():
            parser.error(f"Invalid YOLOv7 repository: missing {repo / required}")

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
                    value = str(Path(value).resolve())
                resolved_args.append(f"{flag}={value}")
            elif index + 1 < len(training_args) and not training_args[index + 1].startswith("-"):
                value = training_args[index + 1]
                resolved_args.extend([flag, str(Path(value).resolve()) if value else value])
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
    if "--help" in command or "-h" in command:
        print("Launcher option: --repo-path PATH (default: sibling yolov7 repository)", flush=True)
        print("File arguments resolve from your current directory; dataset entries resolve from YOLOv7.", flush=True)
    return subprocess.call(command, cwd=repo, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
