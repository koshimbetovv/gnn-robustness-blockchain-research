import torch

from src.models.chronowave_gnn import ChronoWaveGNN


STATIC_MODELS = ("gcn", "graphsage", "gat", "chronowave_gnn")
# CoSemiGNN is per-timestep-slice with 6 auxiliary semi-supervised feature
# columns derived via a cached pipeline — it needs its own attack driver that
# iterates over timesteps and respects the semi-prediction derivation.
TEMPORAL_MODELS = ("recgnn", "evolvegcn_o", "cosemignn")


def forward_logits(model, x, edge_index, *, time_step=None):
    """Normalize heterogeneous model forwards into `(N, num_classes)` logits.

    GCN / GraphSAGE / GAT take `(x, edge_index)` and return `(N, 2)` directly.
    ChronoWaveGNN additionally requires `time_step`.
    CoSemiGNN returns `(out_line, _)` with `out_line` of shape `(N,)` (single-logit,
    BCE-style). We lift to `(N, 2)` as `[0, s]`, which preserves both argmax and
    cross-entropy gradients (CE on `[0, s]` equals BCE-with-logits on `s`).
    """
    if isinstance(model, ChronoWaveGNN):
        if time_step is None:
            raise ValueError("ChronoWaveGNN requires time_step.")
        return model(x, edge_index, time_step)

    if model.__class__.__name__ == "CoSemiGNN":
        raise NotImplementedError(
            "CoSemiGNN uses per-timestep slices and 6 semi-supervised auxiliary "
            "features; it cannot be invoked through the whole-graph static forward "
            "adapter. Use a CoSemi-specific attack driver."
        )

    return model(x, edge_index)
