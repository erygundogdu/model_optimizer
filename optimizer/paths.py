"""Shared path defaults and validation, without training dependencies."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_YOLOV7_REPO = PROJECT_DIR.parent / 'yolov7'


def resolve_yolov7_repo(path=DEFAULT_YOLOV7_REPO):
    repo = Path(path).expanduser().resolve()
    for required in ('models/yolo.py', 'models/experimental.py',
                     'utils/general.py', 'utils/datasets.py', 'test.py'):
        if not (repo / required).is_file():
            raise FileNotFoundError(
                f'Invalid YOLOv7 source directory: missing {repo / required}. '
                'Place a compatible YOLOv7 checkout beside model_optimizer '
                'as yolov7, or pass --repo-path PATH.'
            )
    return repo
