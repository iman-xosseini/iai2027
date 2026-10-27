"""Stage 3 training of Model A: encoder and temporal predictor trained jointly on clean Azure windows.

Stage 4 reuses the same loop for Models B and C through optional hooks of `fit`: a training-batch transform
(the paired corruption) and extra validation sets entering the checkpoint-selection objective. With the
defaults, `fit` behaves exactly as in Stage 3.
"""
import copy
import math
import time

import torch
from torch.utils.data import DataLoader, Subset

from ..models import TemporalWorldModel
from .diagnostics import grad_norm, latent_statistics, noncollapse_checks
from .losses import SIGReg, world_model_loss

LOSS_KEYS = ["prediction_loss", "sigreg_loss", "weighted_sigreg_loss", "total_loss"]
TENSOR_KEYS = ["context", "context_mask", "target", "target_mask"]


def select_device(preference="auto"):
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def to_device(batch, device):
    """The four model inputs of a DataLoader batch, on `device` (metadata stays behind)."""
    return {k: batch[k].to(device, non_blocking=True) for k in TENSOR_KEYS}


def build_model(config, seed, device="cpu"):
    """A freshly initialized world model; the seed fixes the weight initialization."""
    torch.manual_seed(seed)
    return TemporalWorldModel.from_config(config).to(device)


def build_sigreg(config, device="cpu"):
    s = config["training"]["sigreg"]
    return SIGReg(s["num_proj"], s["knots"], s["t_max"]).to(device)


def hyperparameters(config, device):
    t = config["training"]
    return {
        "optimizer": t["optimizer"], "learning_rate": t["learning_rate"], "batch_size": t["batch_size"]["azure"],
        "max_epochs": t["max_epochs"], "sigreg_weight": t["sigreg_weight"], **{f"sigreg_{k}": v for k, v in t["sigreg"].items()},
        **{f"diagnostics_{k}": v for k, v in t["diagnostics"].items()},
        "device": str(device), "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),  # plain str for weights_only loading
    }


def train_one_epoch(model, loader, optimizer, sigreg, sigreg_weight, device, generator, max_batches=None,
                    transform=None):
    """One pass over `loader`. Returns sample-weighted means of the loss components and gradient norms.

    `generator` draws the SIGReg projection directions, so a run is reproducible from its seed.
    `transform` (Stage 4, optional) maps the four model inputs of every batch before the forward pass.
    """
    model.train()
    sums = dict.fromkeys(LOSS_KEYS + ["encoder_grad_norm", "predictor_grad_norm"], 0.0)
    n_seen = n_batches = 0
    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        inputs = to_device(batch, device)                                     # 1. tensors to the device
        if transform is not None:
            inputs = transform(inputs)                                        #    Stage 4: corrupt the history
        optimizer.zero_grad(set_to_none=True)                                 # 2. reset gradients
        outputs = model(**inputs)                                             # 3. Stage 2 forward pass
        losses = world_model_loss(outputs, sigreg, sigreg_weight, generator)  # 4-6. prediction, SIGReg, total
        losses["total_loss"].backward()                                       # 7. backpropagate
        enc_norm, pred_norm = grad_norm(model.encoder), grad_norm(model.predictor)  # 8. validate gradients
        if not (math.isfinite(enc_norm) and math.isfinite(pred_norm) and torch.isfinite(losses["total_loss"])):
            raise FloatingPointError(f"batch {n_batches}: total loss {losses['total_loss'].item()}, "
                                     f"encoder grad norm {enc_norm}, predictor grad norm {pred_norm}")
        optimizer.step()                                                      # 9. update
        n = len(inputs["target"])                                             # 10. record
        for k in LOSS_KEYS:
            sums[k] += losses[k].item() * n
        sums["encoder_grad_norm"] += enc_norm * n
        sums["predictor_grad_norm"] += pred_norm * n
        n_seen += n
        n_batches += 1
    return {**{k: v / n_seen for k, v in sums.items()}, "batches": n_batches}


@torch.no_grad()
def evaluate(model, dataset, sigreg, sigreg_weight, device, batch_size, seed):
    """Losses on `dataset` in evaluation mode, with a reproducible policy.

    Every call uses the same batch composition (a fixed shuffle, so batches mix machines and times as in
    training) and the same SIGReg projection directions, so epochs can be compared directly.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    generator = torch.Generator().manual_seed(seed)
    sums, n_seen = dict.fromkeys(LOSS_KEYS, 0.0), 0
    for batch in loader:
        inputs = to_device(batch, device)
        losses = world_model_loss(model(**inputs), sigreg, sigreg_weight, generator)
        n = len(inputs["target"])
        for k in LOSS_KEYS:
            sums[k] += losses[k].item() * n
        n_seen += n
    return {k: v / n_seen for k, v in sums.items()}


@torch.no_grad()
def encode_windows(model, dataset, indices=None, device="cpu", batch_size=1024):
    """Model outputs for dataset[indices] (all windows if None), in evaluation mode, on the CPU."""
    model.eval()
    subset = dataset if indices is None else Subset(dataset, list(indices))
    parts = {"z_history": [], "z_pred": [], "z_target": []}
    for batch in DataLoader(subset, batch_size=batch_size):
        outputs = model(**to_device(batch, device))
        for k in parts:
            parts[k].append(outputs[k].cpu())
    return {k: torch.cat(v) for k, v in parts.items()}


def _clone_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


MODEL_A_NAME = "Model A: standard temporal LeWorldModel (clean windows, all-ones masks)"
MODEL_A_SELECTION = "lowest validation total loss among epochs passing the non-collapse checks"


def fit(model, train_dataset, val_dataset, config, seed, device, diag_indices, tracked_indices,
        checkpoint_path=None, max_epochs=None, log=print, train_transform=None, extra_val_sets=None,
        selection_weights=None, model_name=MODEL_A_NAME, checkpoint_extra=None):
    """Train `model` and select the checkpoint with the lowest validation objective among non-collapsed epochs.

    Epoch 0 is the untrained model (validation and diagnostics only). After training, `model` holds the
    selected weights; if `checkpoint_path` is given, the selected checkpoint and the full history are saved.
    Returns {"history": [...one dict per epoch...], "best_epoch": int or None, "checkpoint": dict or None}.

    Stage 4 options (the defaults reproduce Stage 3):
    train_transform   : applied to every training batch (`train_one_epoch(transform=...)`); if it has an
                        `epoch_stats()` method, its result is added to the epoch row
    extra_val_sets    : {name: dataset} evaluated like `val_dataset` every epoch, logged as val_<name>_<loss>
    selection_weights : {name: weight} over "clean" (= val_dataset) and the extra sets; the selection objective
                        is the weighted sum of their validation total losses (default {"clean": 1.0})
    The non-collapse checks always use the clean `val_dataset`.
    """
    t_cfg, d_cfg = config["training"], config["training"]["diagnostics"]
    lam, batch_size = t_cfg["sigreg_weight"], t_cfg["batch_size"]["azure"]
    epochs = t_cfg["max_epochs"] if max_epochs is None else max_epochs
    val_seed = t_cfg["sigreg"]["val_projection_seed"]
    extra_val_sets = extra_val_sets or {}
    selection_weights = selection_weights or {"clean": 1.0}
    unknown = set(selection_weights) - {"clean", *extra_val_sets}
    if unknown:
        raise ValueError(f"selection weights for unknown validation sets: {sorted(unknown)}")
    if selection_weights == {"clean": 1.0}:
        selection_rule = MODEL_A_SELECTION
    else:
        terms = " + ".join(f"{w} x val {name} total loss" for name, w in selection_weights.items())
        selection_rule = f"lowest ({terms}) among epochs passing the non-collapse checks (clean validation)"

    sigreg = build_sigreg(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    projection_generator = torch.Generator().manual_seed(seed)  # training SIGReg directions
    tracked_ids = [int(val_dataset[i]["metadata"]["sample_id"]) for i in tracked_indices]

    def record(epoch, train_stats):
        val = evaluate(model, val_dataset, sigreg, lam, device, batch_size, val_seed)
        extra = {name: evaluate(model, ds, sigreg, lam, device, batch_size, val_seed) for name, ds in extra_val_sets.items()}
        totals = {"clean": val["total_loss"], **{name: v["total_loss"] for name, v in extra.items()}}
        stats = latent_statistics(encode_windows(model, val_dataset, diag_indices, device)["z_target"])
        checks = noncollapse_checks(stats, d_cfg["min_effective_rank"], d_cfg["min_latent_std"])
        tracked = encode_windows(model, val_dataset, tracked_indices, device)
        window_mse = (tracked["z_pred"] - tracked["z_target"]).pow(2).mean(dim=1)
        row = {
            "seed": seed, "epoch": epoch,
            **{f"train_{k}": train_stats.get(k, math.nan) for k in LOSS_KEYS},
            **{f"val_{k}": val[k] for k in LOSS_KEYS},
            **{f"val_{name}_{k}": v[k] for name, v in extra.items() for k in LOSS_KEYS},
            "selection_objective": sum(w * totals[name] for name, w in selection_weights.items()),
            "encoder_grad_norm": train_stats.get("encoder_grad_norm", math.nan),
            "predictor_grad_norm": train_stats.get("predictor_grad_norm", math.nan),
            **{k: v for k, v in stats.items() if not isinstance(v, list)},
            **checks, "noncollapsed": all(checks.values()),
            **{f"window_{sid}_mse": float(e) for sid, e in zip(tracked_ids, window_mse)},
        }
        return row, stats

    row, _ = record(0, {})
    row["epoch_time_s"], row["selected"] = 0.0, False
    history, best, best_state = [row], None, None
    log(_format_row(row, False))
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, optimizer, sigreg, lam, device, projection_generator,
                                      transform=train_transform)
        if hasattr(train_transform, "epoch_stats"):
            train_stats.update(train_transform.epoch_stats())
        row, stats = record(epoch, train_stats)
        row.update({k: v for k, v in train_stats.items() if k.startswith("corruption_")})
        row["epoch_time_s"] = time.perf_counter() - start
        is_best = row["noncollapsed"] and (best is None or row["selection_objective"] < best["selection_objective"])
        if is_best:
            best = row
            best_state = {"encoder": _clone_state(model.encoder), "predictor": _clone_state(model.predictor),
                          "optimizer": copy.deepcopy(optimizer.state_dict()), "latent_statistics": stats}
        history.append(row)
        log(_format_row(row, is_best))
    for row in history:
        row["selected"] = best is not None and row["epoch"] == best["epoch"]

    if best is None:
        log("No epoch passed the non-collapse checks: no checkpoint selected.")
        return {"history": history, "best_epoch": None, "checkpoint": None}
    model.encoder.load_state_dict(best_state["encoder"])
    model.predictor.load_state_dict(best_state["predictor"])
    model.eval()
    checkpoint = {
        "model": model_name,
        "encoder": best_state["encoder"], "predictor": best_state["predictor"], "optimizer": best_state["optimizer"],
        "epoch": best["epoch"], "seed": seed, "config": config, "hyperparameters": hyperparameters(config, device),
        "selection": selection_rule,
        "val_metrics": {k: v for k, v in best.items() if k.startswith("val_") or k == "selection_objective"},
        "latent_statistics": best_state["latent_statistics"], "sigreg": sigreg.settings(), "history": history,
        **(checkpoint_extra or {}),
    }
    if checkpoint_path is not None:
        torch.save(checkpoint, checkpoint_path)
        log(f"Saved epoch {best['epoch']} to {checkpoint_path}")
    return {"history": history, "best_epoch": best["epoch"], "checkpoint": checkpoint}


def load_checkpoint(path, device="cpu"):
    """Rebuild the world model from a checkpoint written by `fit`. Returns (model in eval mode, checkpoint dict)."""
    checkpoint = torch.load(path, map_location=device)  # tensors and plain Python values: weights_only=True works
    model = TemporalWorldModel.from_config(checkpoint["config"]).to(device)
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.predictor.load_state_dict(checkpoint["predictor"])
    return model.eval(), checkpoint


def _format_row(row, is_best):
    train = (f"train total {row['train_total_loss']:.4f} (pred {row['train_prediction_loss']:.4f}, "
             f"SIGReg {row['train_sigreg_loss']:.3f})" if row["epoch"] else "untrained" + " " * 38)
    objective = (f" | objective {row['selection_objective']:.4f}"
                 if row["selection_objective"] != row["val_total_loss"] else "")
    return (f"epoch {row['epoch']:2d} | {train} | val total {row['val_total_loss']:.4f} "
            f"(pred {row['val_prediction_loss']:.4f}, SIGReg {row['val_sigreg_loss']:.3f}){objective} | "
            f"eff. rank {row['effective_rank']:4.1f}, min std {row['latent_std_min']:.3f} | "
            f"{row['epoch_time_s']:3.0f} s{'  <- best' if is_best else ''}")
