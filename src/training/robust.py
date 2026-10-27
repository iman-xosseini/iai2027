"""Stage 4 training of Models B and C: the Stage 3 `fit`, with paired training corruption and a mixed selection objective.

Models B and C differ in ONE setting, the input-mask policy ("ones" vs "mask"). For a given seed they share
the initial weights (build_model), the minibatch order (the loader generator in `fit`), the SIGReg directions
(the projection generator in `fit`) and the corruption decisions (TrainingCorruption, a separate stream).

Command line (trains the missing checkpoints; the notebook loads them):
    python -m src.training.robust --models B C --seeds 11 22 33 --threads 4
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..data import (TIME_COLUMNS, TrainingCorruption, corrupted_dataset, load_azure_dataset, make_generator,
                    sample_window_availability)
from .trainer import build_model, fit, hyperparameters, select_device

MODEL_NAMES = {
    "A": "Model A: standard temporal LeWorldModel (clean windows, all-ones masks)",
    "B": "Model B: corruption-augmented temporal LeWorldModel (clean + corrupted windows, all-ones masks)",
    "C": "Model C: mask-aware temporal LeWorldModel (clean + corrupted windows, actual availability masks)",
}


def load_configs(project_root):
    """(base config, Stage 4 config) from configs/base.yaml and configs/stage4.yaml."""
    root = Path(project_root)
    with (root / "configs" / "base.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with (root / "configs" / "stage4.yaml").open(encoding="utf-8") as f:
        stage4 = yaml.safe_load(f)
    return config, stage4


def diagnostic_indices(n_val, diag_cfg):
    """The fixed Stage 3 diagnostic subset and tracked windows (same formula as notebook Section 7.1)."""
    diag = np.linspace(0, n_val - 1, diag_cfg["subset_size"]).round().astype(int).tolist()
    tracked = np.linspace(0, n_val - 1, diag_cfg["tracked_windows"] + 2).round().astype(int)[1:-1].tolist()
    return diag, tracked


def selection_availability(val_dataset, stage4):
    """Fixed availability masks of the corrupted validation set used for checkpoint selection.

    Every window is corrupted, at a rate drawn from `corrupted_val_rates`; the clean half of the selection
    objective is the unmodified validation set. The masks depend only on the configured seed.
    """
    sel = stage4["selection"]
    generator = make_generator(sel["corrupted_val_seed"])
    return sample_window_availability(len(val_dataset), val_dataset.context.shape[1:], sel["corrupted_val_rates"],
                                      0.0, generator)


def train_corrupted_model(name, config, stage4, seed, train_dataset, val_dataset, device, checkpoint_path, log=print):
    """Train Model B or C for one seed with the Stage 3 `fit`; saves and returns the fit result."""
    model_cfg = stage4["models"][name]
    input_mask = model_cfg["input_mask"]
    tc, sel = stage4["training_corruption"], stage4["selection"]
    corruption = TrainingCorruption(input_mask, tc["rates"], tc["clean_fraction"], seed, tc["stream_seed"],
                                    stage4["fill_value"])
    val_availability, val_rates = selection_availability(val_dataset, stage4)
    extra = {"corrupted": corrupted_dataset(val_dataset, val_availability, input_mask, stage4["fill_value"])}
    diag, tracked = diagnostic_indices(len(val_dataset), config["training"]["diagnostics"])
    model = build_model(config, seed, device)
    extra_info = {
        "model_key": name, "input_mask": input_mask, "stage4_config": stage4,
        "training_corruption": corruption.settings(),
        "selection_set": {"rates": sel["corrupted_val_rates"], "seed": sel["corrupted_val_seed"],
                          "removed_fraction": float(1 - val_availability.mean()),
                          "rate_counts": {str(r): int((val_rates == r).sum()) for r in sel["corrupted_val_rates"]}},
        "selection_weights": sel["weights"],
        "random_seeds": {"weight_init": seed, "minibatch_order": seed, "sigreg_projections": seed,
                         "corruption_stream": [tc["stream_seed"], seed],
                         "validation_projections": config["training"]["sigreg"]["val_projection_seed"]},
        "training_settings": hyperparameters(config, device),
    }
    return fit(model, train_dataset, val_dataset, config, seed, device, diag, tracked, checkpoint_path=checkpoint_path,
               log=log, train_transform=corruption, extra_val_sets=extra, selection_weights=sel["weights"],
               model_name=MODEL_NAMES[name], checkpoint_extra=extra_info)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["B", "C"], choices=["B", "C"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (default: torch's choice)")
    parser.add_argument("--retrain", action="store_true", help="train even if the checkpoint exists")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    config, stage4 = load_configs(root)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = select_device(config["training"]["device"])
    metadata = pd.read_csv(root / "data" / "processed" / "window_metadata.csv", parse_dates=TIME_COLUMNS)
    train_ds = load_azure_dataset(root / "data" / "processed", "train", metadata)
    val_ds = load_azure_dataset(root / "data" / "processed", "val", metadata)
    for seed in args.seeds or config["training"]["random_seeds"]:
        for name in args.models:
            path = root / stage4["models"][name]["checkpoint"].format(seed=seed)
            if path.exists() and not args.retrain:
                print(f"Model {name}, seed {seed}: {path.name} exists, skipped", flush=True)
                continue
            print(f"--- Model {name}, seed {seed}, {torch.get_num_threads()} threads " + "-" * 60, flush=True)
            start = time.perf_counter()
            path.parent.mkdir(parents=True, exist_ok=True)
            result = train_corrupted_model(name, config, stage4, seed, train_ds, val_ds, device, path,
                                           log=lambda line: print(line, flush=True))
            print(f"Model {name}, seed {seed}: selected epoch {result['best_epoch']} "
                  f"in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
