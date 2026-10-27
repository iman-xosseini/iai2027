"""Stage 5 final held-out evaluation: frozen protocol, provenance hashes and evaluation-only scoring.

Nothing here trains: models are put in evaluation mode, predictions run under torch.inference_mode(), and no
optimizer is ever constructed. Parameter fingerprints taken before and after scoring prove the weights did
not change.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from .robustness import predict


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(tensor):
    """SHA-256 of a tensor's raw bytes (dtype and shape are part of the caller's record, not of the hash)."""
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def state_fingerprint(model):
    """SHA-256 over all parameters and buffers, in state-dict order."""
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def freeze_protocol(config_path, manifest_path, guard_paths=(), extra=None):
    """Write (first call) or verify (later calls) the protocol manifest: SHA-256 and timestamp of the config.

    On the first call, none of `guard_paths` (e.g. the per-window test scores) may exist yet, which records
    that the protocol was frozen before any test prediction. Later calls raise if the config changed; a
    deliberate correction must be logged as an amendment instead (see `record_amendment`).
    """
    config_path, manifest_path = Path(config_path), Path(manifest_path)
    digest = file_sha256(config_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = manifest["amendments"][-1]["sha256"] if manifest.get("amendments") else manifest["sha256"]
        if digest != current:
            raise RuntimeError(f"{config_path.name} changed after freezing (sha256 {digest[:12]} != {current[:12]}); "
                               "log the change with record_amendment() instead of overwriting the protocol")
        manifest["verified_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return manifest, False
    existing = [str(p) for p in guard_paths if Path(p).exists()]
    if existing:
        raise RuntimeError(f"cannot freeze: evaluation outputs already exist: {existing}")
    manifest = {"config": config_path.as_posix(), "sha256": digest,
                "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "test_outputs_existed_at_freeze": False, "amendments": [], **(extra or {})}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, True


def record_amendment(config_path, manifest_path, reason, test_results_inspected):
    """Log a deliberate post-freeze change of the protocol, with the new hash and whether test results were seen."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest.setdefault("amendments", []).append({
        "sha256": file_sha256(config_path), "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason, "test_results_inspected_before_change": bool(test_results_inspected)})
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def evaluate_only(model, inputs, device="cpu", batch_size=4096):
    """z_pred and z_target for full-partition inputs, in evaluation mode and under torch.inference_mode()."""
    model.eval()
    with torch.inference_mode():
        return predict(model, inputs, device, batch_size)
