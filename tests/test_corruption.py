"""Unit tests for src/data/corruption.py (Stage 4).

Run with `python -m pytest tests` or, without pytest, `python tests/test_corruption.py`.
The notebook (Section 8.2) runs the same functions through `run_all()`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.corruption import (TrainingCorruption, WindowTimeline, apply_dropout, burst_schedule, check_invariants,
                                 independent_mask, make_generator, outage_schedule, policy_inputs,
                                 sample_window_availability, scenario_schedule, stream_seed)

SHAPE = (4, 6, 4)  # context blocks x hours x sensors


def _batch(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"context": torch.randn(n, *SHAPE, generator=g), "context_mask": torch.ones(n, *SHAPE),
            "target": torch.randn(n, 6, 4, generator=g), "target_mask": torch.ones(n, 6, 4)}


def _timeline(machines=3, windows=40, stride=6):
    rows = [{"machine_id": m + 1, "context_start": pd.Timestamp("2015-09-13 18:00") + pd.Timedelta(hours=stride * w)}
            for m in range(machines) for w in range(windows)]
    return WindowTimeline(pd.DataFrame(rows))


def test_zero_rate_keeps_everything():
    batch = _batch()
    availability = independent_mask(batch["context"].shape, 0.0, make_generator(1))
    out = policy_inputs(batch["context"], availability, batch["target"], "mask")
    assert (availability == 1).all()
    assert torch.equal(out["context"], batch["context"]) and (out["context_mask"] == 1).all()


def test_independent_rates_match_requested_probabilities():
    for rate in (0.1, 0.2, 0.4):
        availability = independent_mask((20000, *SHAPE), rate, make_generator(7, int(rate * 100)))
        observed = 1 - availability.mean().item()
        # 1.92M Bernoulli draws: standard error < 4e-4, so 2e-3 is a > 5-sigma tolerance
        assert abs(observed - rate) < 2e-3, (rate, observed)


def test_invariants_hold_for_both_policies():
    batch = _batch()
    original = batch["context"].clone()
    availability = independent_mask(batch["context"].shape, 0.3, make_generator(2))
    for policy in ("ones", "mask"):
        out = policy_inputs(batch["context"], availability, batch["target"], policy)
        checks = check_invariants(batch, out, availability, original_context=original)
        assert all(checks.values()), {k: v for k, v in checks.items() if not v}
        expected_mask = torch.ones_like(availability) if policy == "ones" else availability
        assert torch.equal(out["context_mask"], expected_mask)
    assert torch.equal(batch["context"], original)


def test_policies_differ_only_in_mask_channels():
    from src.models import mask_conditioned_input
    batch = _batch()
    availability = independent_mask(batch["context"].shape, 0.4, make_generator(3))
    b = policy_inputs(batch["context"], availability, batch["target"], "ones")
    c = policy_inputs(batch["context"], availability, batch["target"], "mask")
    assert torch.equal(b["context"], c["context"]) and torch.equal(b["target"], c["target"])
    u_b, u_c = mask_conditioned_input(b["context"], b["context_mask"]), mask_conditioned_input(c["context"], c["context_mask"])
    assert torch.equal(u_b[..., :24], u_c[..., :24])         # identical value halves of the encoder input
    assert not torch.equal(u_b[..., 24:], u_c[..., 24:])     # only the mask half differs


def test_seeds_reproduce_and_differ():
    assert stream_seed(4001, 11) == stream_seed(4001, 11) != stream_seed(4001, 22)
    assert stream_seed("burst_6h", 1) == stream_seed("burst_6h", 1)  # strings: CRC-32, not salted hash()
    a = independent_mask((64, *SHAPE), 0.2, make_generator(5))
    b = independent_mask((64, *SHAPE), 0.2, make_generator(5))
    c = independent_mask((64, *SHAPE), 0.2, make_generator(6))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_training_corruption_pairs_b_and_c():
    batch = _batch(128)
    b = TrainingCorruption("ones", [0.1, 0.2, 0.4], 0.5, seed=11, stream=4001)
    c = TrainingCorruption("mask", [0.1, 0.2, 0.4], 0.5, seed=11, stream=4001)
    for _ in range(5):
        ob, oc = b(batch), c(batch)
        assert torch.equal(ob["context"], oc["context"])
        assert (ob["context_mask"] == 1).all()
        removed = oc["context_mask"] == 0
        assert (oc["context"][removed] == 0).all()
    sb, sc = b.epoch_stats(), c.epoch_stats()
    assert sb == sc and sb["corruption_corrupted_fraction"] == 0.5
    other = TrainingCorruption("mask", [0.1, 0.2, 0.4], 0.5, seed=22, stream=4001)
    assert not torch.equal(other(batch)["context_mask"], c(batch)["context_mask"])


def test_training_mixture_composition():
    availability, rate = sample_window_availability(30000, SHAPE, [0.1, 0.2, 0.4], 0.5, make_generator(9))
    assert (rate == 0).sum() == 15000
    clean = availability[rate == 0]
    assert (clean == 1).all()
    for r in (0.1, 0.2, 0.4):
        share = (rate == r).float().mean().item()
        assert abs(share - 1 / 6) < 0.01, (r, share)
        observed = 1 - availability[rate == r].mean().item()
        assert abs(observed - r) < 5e-3, (r, observed)


def test_apply_dropout_never_writes_in_place():
    context = torch.randn(8, *SHAPE)
    original = context.clone()
    availability = torch.zeros_like(context)
    out = apply_dropout(context, availability)
    assert (out == 0).all() and torch.equal(context, original)


def test_timeline_windows_agree_on_shared_hours():
    tl = _timeline()
    schedule = burst_schedule(tl.num_machines, tl.hours, 4, duration=12, cycle=30, generator=make_generator(4))
    masks = tl.window_masks(schedule)                     # [N, 4, 6, 4]
    # Consecutive windows of one machine share three blocks: block k + 1 of window i == block k of window i + 1.
    same_machine = tl.machine_index[1:] == tl.machine_index[:-1]
    assert torch.equal(masks[1:][same_machine][:, :3], masks[:-1][same_machine][:, 1:])
    assert ((masks == 0) | (masks == 1)).all()


def test_burst_is_one_sensor_and_consecutive():
    hours, cycle = 300, 30
    for duration in (6, 12):
        s = burst_schedule(2, hours, 4, duration, cycle, make_generator(duration))
        for m in range(2):
            for c in range(hours // cycle):
                period = s[m, c * cycle:(c + 1) * cycle]          # [30, 4]
                missing = (period == 0)
                assert missing.sum() == duration
                sensors = missing.any(0).nonzero().flatten()
                assert len(sensors) == 1
                hrs = missing[:, sensors[0]].nonzero().flatten()
                assert (hrs.diff() == 1).all()


def test_outage_is_block_aligned():
    hours = 300
    s = outage_schedule(3, hours, 4, block_size=6, blocks=[1, 2], cycle_blocks=5, generator=make_generator(8))
    lengths = []
    for m in range(3):
        for c in range(hours // 30):
            missing = s[m, c * 30:(c + 1) * 30] == 0
            sensors = missing.any(0).nonzero().flatten()
            assert len(sensors) == 1
            hrs = missing[:, sensors[0]].nonzero().flatten()
            assert hrs[0] % 6 == 0 and len(hrs) % 6 == 0 and (hrs.diff() == 1).all()
            lengths.append(len(hrs) // 6)
    assert set(lengths) == {1, 2}


def test_scenario_schedules_reproducible():
    tl = _timeline()
    spec = {"pattern": "outage", "blocks": [1, 2], "cycle_blocks": 5}
    a = scenario_schedule(spec, tl, 4, make_generator(4201, "outage"))
    b = scenario_schedule(spec, tl, 4, make_generator(4201, "outage"))
    assert torch.equal(a, b)
    spec = {"pattern": "independent", "rate": 0.2}
    a = scenario_schedule(spec, tl, 4, make_generator(4201, "independent_20"))
    b = scenario_schedule(spec, tl, 4, make_generator(4201, "independent_20"))
    assert torch.equal(a, b) and abs((1 - a.mean().item()) - 0.2) < 0.02


TESTS = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]


def run_all():
    """Run every test; returns a DataFrame-ready list of (test, passed, message)."""
    results = []
    for test in TESTS:
        try:
            test()
            results.append((test.__name__, True, ""))
        except Exception as err:  # report every failure instead of stopping at the first
            results.append((test.__name__, False, f"{type(err).__name__}: {err}"))
    return results


if __name__ == "__main__":
    outcome = run_all()
    for name, ok, msg in outcome:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}")
    sys.exit(0 if all(ok for _, ok, _ in outcome) else 1)
