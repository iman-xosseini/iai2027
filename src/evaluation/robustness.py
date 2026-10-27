"""Stage 4 robustness metrics: paired clean/corrupted prediction errors, surprise scores, alerts, temporal checks.

Every comparison is made WITHIN one trained model: its clean and corrupted predictions are scored against the
same target embedding (the target is never corrupted), so ratios are comparable across separately trained
latent spaces while absolute MSE values are not.
"""
import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def predict(model, inputs, device="cpu", batch_size=4096):
    """z_pred and z_target [N, 32] on the CPU for full-partition input tensors {context, context_mask, target, target_mask}."""
    model.eval()
    n = len(inputs["target"])
    parts = {"z_pred": [], "z_target": []}
    for start in range(0, n, batch_size):
        out = model(**{k: v[start:start + batch_size].to(device) for k, v in inputs.items()})
        for k in parts:
            parts[k].append(out[k].cpu())
    return {k: torch.cat(v) for k, v in parts.items()}


def window_scores(outputs):
    """Per-window prediction MSE (mean over the 32 dimensions, the Stage 3 reduction) and surprise ||z_pred - z_target||_2."""
    diff = outputs["z_pred"] - outputs["z_target"]
    return {"mse": diff.pow(2).mean(dim=1), "surprise": diff.norm(dim=1)}


def affected_windows(availability):
    """Windows with at least one removed historical reading: availability [N, ...] -> bool [N]."""
    return (availability.reshape(len(availability), -1) == 0).any(dim=1)


def paired_metrics(clean, corrupted, affected):
    """Clean-vs-corrupted metrics of one model on identical windows (window_scores dicts, bool [N] affected)."""
    mse_clean, mse_corr = clean["mse"].mean().item(), corrupted["mse"].mean().item()
    change = corrupted["mse"] - clean["mse"]
    s_change = corrupted["surprise"] - clean["surprise"]
    has_affected = bool(affected.any())
    return {
        "clean_mse": mse_clean,
        "corrupted_mse": mse_corr,
        "relative_mse": mse_corr / mse_clean,
        "relative_degradation": mse_corr / mse_clean - 1,
        "affected_fraction": affected.float().mean().item(),
        "relative_mse_affected": (corrupted["mse"][affected].mean() / clean["mse"][affected].mean()).item()
                                 if has_affected else 1.0,
        "window_error_change_mean": change.mean().item(),
        "window_error_change_median": change.median().item(),
        "window_error_change_p90": change.quantile(0.9).item(),
        "windows_error_increased_affected": (change[affected] > 0).float().mean().item() if has_affected else float("nan"),
        "clean_surprise_mean": clean["surprise"].mean().item(),
        "corrupted_surprise_mean": corrupted["surprise"].mean().item(),
        "surprise_change_mean": s_change.mean().item(),
        "relative_surprise_change": (s_change.mean() / clean["surprise"].mean()).item(),
    }


def alert_threshold(clean_surprise, percentile):
    """Model-specific alert threshold: a percentile of the clean-validation surprise distribution."""
    return float(np.percentile(clean_surprise.numpy(), percentile))


def alert_rate(surprise, threshold, subset=None):
    """Fraction of windows (optionally of a bool subset) whose surprise exceeds the fixed threshold."""
    exceeds = surprise > threshold
    if subset is not None:
        exceeds = exceeds[torch.as_tensor(subset)]
    return exceeds.float().mean().item()


# --- Machine-level bootstrap --------------------------------------------------------------------------------------
# Windows of one machine overlap and share its operating regime, so they are not independent observations.
# Uncertainty over machines is estimated by resampling whole machines: replicate r gives machine m the weight
# w[r, m] = number of times it was drawn, and every ratio of window sums is recomputed with those weights.

def bootstrap_weights(num_machines, replicates, seed):
    """[replicates, num_machines] multinomial resampling counts (each row sums to num_machines)."""
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(num_machines, (replicates, num_machines), generator=generator)
    return torch.zeros(replicates, num_machines, dtype=torch.float64).scatter_add_(
        1, draws, torch.ones_like(draws, dtype=torch.float64))


def machine_sums(values, machine_index, num_machines):
    """Sum of a per-window quantity [N] within each machine -> [num_machines] (float64)."""
    out = torch.zeros(num_machines, dtype=torch.float64)
    return out.index_add_(0, torch.as_tensor(machine_index), torch.as_tensor(values, dtype=torch.float64))


def weighted_ratio(numerator_sums, denominator_sums, weights):
    """Ratio of machine-weighted sums for every bootstrap replicate: [replicates]."""
    return (weights @ numerator_sums) / (weights @ denominator_sums)


def interval(samples, level=0.95):
    lo, hi = np.quantile(np.asarray(samples), [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


# --- Failure-negative proxy subset ---------------------------------------------------------------------------------

def proxy_normal_windows(metadata, failures, maintenance, rule):
    """bool [N]: windows with no failure in [context_start - before, target_end + after] and no maintenance in
    [context_start - before, target_end + after] for the same machine (hours from the `proxy_normal` rule).

    Labels only select evaluation windows; they never enter training, thresholds or model selection.
    """
    keep = np.ones(len(metadata), dtype=bool)
    md = metadata.reset_index(drop=True)
    for events, before, after in ((failures, rule["failure_before_hours"], rule["failure_after_hours"]),
                                  (maintenance, rule["maintenance_before_hours"], rule["maintenance_after_hours"])):
        times = pd.to_datetime(events["datetime"])
        for machine, rows in md.groupby("machine_id"):
            t = times[events["machine_id"].to_numpy() == machine].to_numpy()
            if len(t) == 0:
                continue
            lo = (pd.to_datetime(rows["context_start"]) - pd.Timedelta(hours=before)).to_numpy()[:, None]
            hi = (pd.to_datetime(rows["target_end"]) + pd.Timedelta(hours=after)).to_numpy()[:, None]
            hit = ((t[None, :] >= lo) & (t[None, :] <= hi)).any(axis=1)
            keep[rows.index.to_numpy()[hit]] = False
    return keep


# --- Temporal-information checks -----------------------------------------------------------------------------------

def block_permutations(n, kind, generator, context_length=4):
    """[n, context_length] block orders: "reversed" (newest first) or "shuffled" (random, never the identity)."""
    if kind == "reversed":
        return torch.arange(context_length - 1, -1, -1).expand(n, -1).clone()
    if kind != "shuffled":
        raise ValueError(f"unknown order perturbation {kind!r}")
    perms = torch.rand(n, context_length, generator=generator).argsort(dim=1)
    identity = (perms == torch.arange(context_length)).all(dim=1)
    while identity.any():  # redraw the identity rows only
        perms[identity] = torch.rand(int(identity.sum()), context_length, generator=generator).argsort(dim=1)
        identity = (perms == torch.arange(context_length)).all(dim=1)
    return perms


def permute_blocks(tensor, perms):
    """Reorder the historical blocks of [N, 4, 6, 4] along dim 1 with per-window permutations [N, 4]."""
    index = perms.view(*perms.shape, 1, 1).expand_as(tensor)
    return tensor.gather(1, index)


def mismatched_history_indices(timeline, min_separation, generator):
    """For every window, another window of the SAME machine at least `min_separation` windows away.

    With a 6 h stride, a separation of s windows shifts the context by 6 s hours; s >= 5 already keeps the
    replacement context outside the original target block, and larger s also removes shared history.
    """
    position, machine = torch.as_tensor(timeline.position), torch.as_tensor(timeline.machine_index)
    order = np.lexsort((timeline.position, timeline.machine_index))  # window index by (machine, position)
    first = np.searchsorted(timeline.machine_index[order], np.arange(timeline.num_machines))
    counts = np.bincount(timeline.machine_index, minlength=timeline.num_machines)
    out = torch.empty(len(position), dtype=torch.long)
    u = torch.rand(len(position), generator=generator)
    for i in range(len(position)):
        m, p, n = int(machine[i]), int(position[i]), int(counts[machine[i]])
        valid = np.r_[0:max(p - min_separation + 1, 0), min(p + min_separation, n):n]
        if len(valid) == 0:
            raise ValueError(f"machine {m} has no window {min_separation} positions away from {p}")
        out[i] = int(order[first[m] + valid[int(u[i] * len(valid))]])
    return out
