import sys
import copy
import torch
from pathlib import Path

from optimizer.adapters.base import ModelAdapter
from optimizer.core.model_bundle import ModelBundle


class YOLOv7Adapter(ModelAdapter):

    def __init__(
        self,
        repo_path: str
    ):
        self.repo_path = Path(repo_path).resolve()

        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"YOLOv7 repo not found: {self.repo_path}"
            )

    def _make_yolov7_importable(self):

        repo = str(self.repo_path)

        if repo not in sys.path:
            sys.path.insert(0, repo)

    def load(
        self,
        weights_path: str,
        device: str = "cpu"
    ) -> ModelBundle:

        self._make_yolov7_importable()

        weights_path = Path(weights_path).resolve()

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {weights_path}"
            )

        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=False
        )

        if isinstance(checkpoint, dict):

            if checkpoint.get("ema") is not None:
                model = checkpoint["ema"]

            elif checkpoint.get("model") is not None:
                model = checkpoint["model"]

            else:
                raise RuntimeError(
                    "YOLOv7 checkpoint contains neither model nor ema."
                )

        else:
            model = checkpoint

        model = model.float().to(device)
        model.eval()

        ignored_layers = self._get_pruning_ignored_layers(
            model
        )

        return ModelBundle(
            model=model,
            provider="yolov7",
            original_checkpoint=checkpoint,
            ignored_pruning_layers=ignored_layers,
            metadata={
                "weights_path": str(weights_path),
                "repo_path": str(self.repo_path),
            }
        )

    def _get_pruning_ignored_layers(
        self,
        model,
        idx= None
    ):
        """
        YOLOv7-specific knowledge belongs here.

        Detection head should not have its semantic output
        channels directly pruned.
        """

        if not hasattr(model, "model"):
            raise RuntimeError(
                "Loaded YOLOv7 model has no model.model structure."
            )

        detect_head = model.model[-1]
        lst = list(detect_head.modules())
        if idx is not None:
            for i in idx:
                lst.extend(model.model[i].modules())
        return lst

    def save(
        self,
        bundle: ModelBundle,
        output_path: str
    ):

        output_path = Path(
            output_path
        ).resolve()

        model = bundle.model.cpu()
        model.eval()

        original = bundle.original_checkpoint

        if isinstance(original, dict):

            checkpoint = copy.copy(original)

            checkpoint["model"] = model

            # These belong to the old architecture.
            checkpoint["ema"] = None
            checkpoint["optimizer"] = None

        else:

            checkpoint = {
                "model": model,
                "ema": None,
                "optimizer": None,
            }

        torch.save(
            checkpoint,
            output_path
        )

        print(
            f"Saved optimized checkpoint to:\n"
            f"{output_path}"
        )

        