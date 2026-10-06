import torch

from optimizer.adapters.base import ModelAdapter
from optimizer.pruning.channel_pruning import (
    apply_gamma_channel_pruning,
)


def run_gamma_pruning_pipeline(
    adapter: ModelAdapter,

    weights_path: str,

    output_path: str,

    input_shape: tuple,

    pruning_ratio: float,

    device: str,
):

    # ==============================================
    # 1. Load
    # ==============================================

    print("\n[1] Loading model")

    bundle = adapter.load(
        weights_path=weights_path,
        device=device,
    )

    model = bundle.model

    # ==============================================
    # 2. Verify original
    # ==============================================

    print("\n[2] Verifying original model")

    dummy = torch.randn(
        *input_shape,
        device=device
    )

    model.eval()

    with torch.no_grad():
        _ = model(dummy)

    print(
        "Original forward pass: OK"
    )

    # ==============================================
    # 3. Optimize
    # ==============================================

    print("\n[3] Applying pruning")

    bundle = apply_gamma_channel_pruning(
        bundle=bundle,
        input_shape=input_shape,
        pruning_ratio=pruning_ratio,
        device=device,
        round_to=None,
    )

    # ==============================================
    # 4. Verify optimized model
    # ==============================================

    print("\n[4] Verifying pruned model")

    bundle.model.eval()

    with torch.no_grad():
        _ = bundle.model(dummy)

    print(
        "Pruned forward pass: OK"
    )

    # ==============================================
    # 5. Save
    # ==============================================

    print("\n[5] Saving")

    adapter.save(
        bundle=bundle,
        output_path=output_path,
    )

    return bundle
