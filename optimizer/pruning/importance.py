import torch
import torch.nn as nn


def collect_bn_gamma(
    model
):
    """
    Collect absolute BN scaling parameters:

        significance_i = |gamma_i|

    gamma = BatchNorm2d.weight
    """

    values = []

    for module in model.modules():

        if isinstance(
            module,
            nn.BatchNorm2d
        ) and module.weight is not None:
            values.append(
                module
                .weight
                .detach()
                .abs()
                .cpu()
            )

    if not values:
        raise RuntimeError(
            "No BatchNorm2d layers found."
        )

    return torch.cat(values)


def print_bn_gamma_statistics(
    model
):

    gamma = collect_bn_gamma(
        model
    )

    print("\nBN gamma statistics")
    print("-------------------")

    print(
        f"Channels : {gamma.numel()}"
    )

    print(
        f"Min      : {gamma.min().item():.6f}"
    )

    print(
        f"Mean     : {gamma.mean().item():.6f}"
    )

    print(
        f"Median   : {gamma.median().item():.6f}"
    )

    print(
        f"Max      : {gamma.max().item():.6f}"
    )
