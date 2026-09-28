"""Training throughput utilities; importing this module never starts CUDA work."""

from collections import defaultdict

import torch


def check_finite(tensor, message):
    finite = torch.isfinite(tensor).all()
    if tensor.is_cuda:
        # Avoid a host synchronization for every loss and gradient check.
        torch._assert_async(finite, message)
    elif not bool(finite):
        raise FloatingPointError(message)


def accumulation_fraction(step, batch_size, accumulation, sample_count):
    """Weight a mean-per-image loss correctly even for a short last batch."""
    group_start = (step // accumulation) * accumulation * batch_size
    group_samples = min(batch_size * accumulation, sample_count - group_start)
    batch_samples = min(batch_size, sample_count - step * batch_size)
    return batch_samples / group_samples


def map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {key: map_tensors(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [map_tensors(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(map_tensors(item, fn) for item in value)
    return value


class CUDAPrefetcher:
    def __init__(self, loader, channels_last=False):
        self.loader = loader
        self.channels_last = channels_last

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        stream = torch.cuda.Stream()
        iterator = iter(self.loader)

        def move(tensor):
            if self.channels_last and tensor.ndim == 4:
                return tensor.to(
                    "cuda", non_blocking=True, memory_format=torch.channels_last
                )
            return tensor.to("cuda", non_blocking=True)

        def load_next():
            try:
                batch = next(iterator)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                return map_tensors(batch, move)

        upcoming = load_next()
        while upcoming is not None:
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(stream)
            current = upcoming
            map_tensors(
                current,
                lambda t, active_stream=current_stream: t.record_stream(active_stream),
            )
            upcoming = load_next()
            yield current


class EMAUpdater:
    """Precompute state references and batch the EMA arithmetic into foreach ops."""

    def __init__(self, ema, model):
        grouped = defaultdict(lambda: ([], []))
        self.integer_pairs = []
        current = model.state_dict()
        for name, target in ema.state_dict().items():
            source = current[name]
            if target.is_floating_point():
                a, b = grouped[(target.device, target.dtype)]
                a.append(target)
                b.append(source)
            else:
                self.integer_pairs.append((target, source))
        self.groups = list(grouped.values())

    @torch.no_grad()
    def update(self, decay):
        for targets, sources in self.groups:
            torch._foreach_lerp_(targets, sources, 1 - decay)
        for target, source in self.integer_pairs:
            target.copy_(source)


def cpu_state_dict(model):
    # Deep CPU snapshot: asynchronous scoring must not later save mutated weights.
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def validate_initialization_config(previous, current):
    if previous["manifest_sha256"] != current["manifest_sha256"]:
        raise ValueError("Initialization checkpoint has a different manifest")
    deployment = current.get("all_data", False) and current["fold"] == -1
    if previous["fold"] != current["fold"] and not deployment:
        raise ValueError("Initialization checkpoint has a different validation fold")


def validate_resume_config(previous, current):
    for key in (
        "kind",
        "fold",
        "size",
        "epochs",
        "lr",
        "seed",
        "manifest_sha256",
        "ema",
        "limit",
    ):
        if previous[key] != current[key]:
            raise ValueError(f"Resume config mismatch: {key}")
    if previous.get("sparse_points", False) != current.get("sparse_points", False):
        raise ValueError("Resume config mismatch: sparse_points")
    if previous.get("dense_masks", False) != current.get("dense_masks", False):
        raise ValueError("Resume config mismatch: dense_masks")
    if previous.get("highres_masks", False) != current.get("highres_masks", False):
        raise ValueError("Resume config mismatch: highres_masks")
    if previous.get("component_balanced", False) != current.get(
        "component_balanced", False
    ):
        raise ValueError("Resume config mismatch: component_balanced")
    if previous.get("annotation_weighted", False) != current.get(
        "annotation_weighted", False
    ):
        raise ValueError("Resume config mismatch: annotation_weighted")
    if previous.get("contrast_input", False) != current.get("contrast_input", False):
        raise ValueError("Resume config mismatch: contrast_input")
    for key, default in [("tile_size", 0), ("train_repeat", 1)]:
        if previous.get(key, default) != current.get(key, default):
            raise ValueError(f"Resume config mismatch: {key}")
    if previous.get("encoder", "tu-convnext_tiny.fb_in22k_ft_in1k") != current.get(
        "encoder", "tu-convnext_tiny.fb_in22k_ft_in1k"
    ):
        raise ValueError("Resume config mismatch: encoder")
    if previous["batch"] * previous["accum"] != current["batch"] * current["accum"]:
        raise ValueError(
            "Resume must preserve effective batch size and optimizer schedule"
        )
