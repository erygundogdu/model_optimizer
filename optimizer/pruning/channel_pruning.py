import copy
import math

import torch
import torch_pruning as tp

from optimizer.core.model_bundle import ModelBundle
from optimizer.pruning.importance import print_bn_gamma_statistics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def print_conv_shapes(model, title):
    print(f'\n{title}')
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f'{name:30s} {module.in_channels} -> {module.out_channels}')


def apply_gamma_channel_pruning(
    bundle: ModelBundle,
    input_shape: tuple,
    pruning_ratio: float,
    device: str = 'cpu',
    round_to: int | None = None,
) -> ModelBundle:
    """Rank abs(BN.weight) globally; propagate selected removals with DepGraph.

    Coupled BN layers share a group whose root BN supplies the scores.
    Safety constraints and optional alignment may reduce the achieved ratio.
    Returns a new bundle without changing the input model.
    """
    if not 0.0 < pruning_ratio < 1.0:
        raise ValueError('pruning_ratio must be between 0 and 1.')
    if round_to is not None and (isinstance(round_to, bool)
                                or not isinstance(round_to, int) or round_to < 1):
        raise ValueError('round_to must be a positive integer or None.')

    # Copy together to preserve ignored-layer references into the new model.
    model, ignored_layers, metadata = copy.deepcopy(
        (bundle.model, bundle.ignored_pruning_layers, bundle.metadata)
    )
    result = ModelBundle(model=model, provider=bundle.provider,
                         original_checkpoint=bundle.original_checkpoint,
                         ignored_pruning_layers=ignored_layers, metadata=metadata)
    model.to(device).eval()
    print_bn_gamma_statistics(model)
    params_before = count_parameters(model)
    example_input = torch.randn(*input_shape, device=device,
                                dtype=next(model.parameters()).dtype)
    ignored = {child for layer in ignored_layers for child in layer.modules()}
    names = {module: name for name, module in model.named_modules()}
    requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        with torch.enable_grad():
            graph = tp.DependencyGraph().build_dependency(model, example_inputs=example_input)
        roots, candidates = [], []
        for group in graph.get_all_groups(
            ignored_layers=list(ignored), root_module_types=(torch.nn.BatchNorm2d,)
        ):
            bn = group[0].dep.target.module
            if bn.weight is None:
                continue
            if any(dep.target.module in ignored
                   and graph.is_out_channel_pruning_fn(dep.handler) for dep, _ in group):
                continue
            gamma = bn.weight.detach().abs().cpu()
            if not torch.isfinite(gamma).all():
                raise ValueError(f'Nonfinite BN gamma in {names[bn]}.')
            root_id = len(roots)
            roots.append(bn)
            candidates.extend((float(value), root_id, index)
                              for index, value in enumerate(gamma.tolist()))
        if not candidates:
            raise RuntimeError('No prunable affine BatchNorm2d groups found.')
        candidates.sort()
        target = math.floor(len(candidates) * pruning_ratio)
        selected = [[] for _ in roots]
        for _, root_id, index in candidates[:target]:
            selected[root_id].append(index)
        records = []
        for bn, indices in zip(roots, selected):
            keep = max(1, bn.num_features - len(indices))
            if round_to is not None:
                keep = min(bn.num_features, math.ceil(keep / round_to) * round_to)
            indices = sorted(indices[:bn.num_features - keep])
            if not indices:
                continue
            group = graph.get_pruning_group(bn, tp.prune_batchnorm_out_channels, idxs=indices)
            if not graph.check_pruning_group(group):
                continue
            if any(dep.target.module in ignored
                   and graph.is_out_channel_pruning_fn(dep.handler) for dep, _ in group):
                continue
            group.prune()
            records.append({'layer': names[bn], 'indices': indices})
        with torch.no_grad():
            model(example_input)
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad[name])
    params_after = count_parameters(model)
    print(f'\nParameters: {params_before:,} -> {params_after:,}')
    result.metadata['pruning'] = {
        'method': 'bn_gamma', 'selection': 'global_abs_bn_weight',
        'ratio': pruning_ratio, 'round_to': round_to,
        'eligible_channels': len(candidates), 'selected_channels': target,
        'pruned_root_channels': sum(len(record['indices']) for record in records),
        'layers': records, 'parameters_before': params_before,
        'parameters_after': params_after,
    }
    return result
