"""Stage 5 protocol tests: event logic, timing causality, alarm grouping, bootstrap clustering, protocol freezing,
and data/checkpoint-level guarantees of the final evaluation.

Run with `python -m pytest tests` or `python tests/test_stage5_protocol.py`; notebook 05 runs `run_all()`.
The data-level tests read the test partition's INPUT tensors (never model outputs on them) and validation scores.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (TIME_COLUMNS, WindowTimeline, load_azure_dataset, make_generator, policy_inputs,
                      scenario_schedule)
from src.evaluation import (alarm_incidents, alert_threshold, bootstrap_weights, credit_windows, detect_episodes,
                            evaluate_only, failure_episodes, failure_events, freeze_protocol, machine_sums,
                            paired_metrics, state_fingerprint, weighted_ratio)
from src.training import load_checkpoint

T0 = pd.Timestamp("2015-11-10 06:00")


def _windows(machine=1, n=12, first_end=pd.Timestamp("2015-11-08 11:00")):
    return pd.DataFrame({"machine_id": [machine] * n, "target_end": [first_end + pd.Timedelta(hours=6 * i) for i in range(n)]})


# --- event grouping and timing --------------------------------------------------------------------------------------

def test_event_grouping_counts_multicomponent_failures_once():
    rows = pd.DataFrame({"datetime": [T0, T0, T0 + pd.Timedelta(days=5)], "machine_id": [1, 1, 1],
                         "component": ["comp1", "comp2", "comp3"], "partition": ["test"] * 3})
    events = failure_events(rows, "test")
    assert len(events) == 2 and events.loc[0, "rows"] == 2 and events.loc[0, "components"] == "comp1,comp2"


def test_episode_gap_merges_close_events():
    events = pd.DataFrame({"machine_id": [1, 1, 1, 2], "datetime": [T0, T0 + pd.Timedelta(hours=12),
                           T0 + pd.Timedelta(hours=60), T0], "components": ["a", "b", "c", "d"], "rows": [1, 1, 1, 1]})
    ep = failure_episodes(events, gap_hours=24)
    assert len(ep) == 3
    first = ep[(ep.machine_id == 1)].iloc[0]
    assert first.failure_time == T0 and first.events == 2


def test_lead_time_causality():
    # windows of machine 1 with target_end every 6 h; failure at T0 = 06:00
    w = _windows()
    ep = pd.DataFrame({"episode": [0], "machine_id": [1], "failure_time": [T0]})
    credit = credit_windows(w, ep, horizon_hours=24)
    lead = (T0 - pd.to_datetime(w["target_end"])) / pd.Timedelta(hours=1)
    assert ((credit == 0) == ((lead > 0) & (lead <= 24))).all()      # 0 < f - target_end <= 24 h only
    # a window ending exactly at f, or after f, is never credited
    w2 = pd.DataFrame({"machine_id": [1, 1], "target_end": [T0, T0 + pd.Timedelta(hours=1)]})
    assert (credit_windows(w2, ep, 24) == -1).all()
    # the 1 h availability offset moves the alarm time later
    w3 = pd.DataFrame({"machine_id": [1], "target_end": [T0 - pd.Timedelta(hours=1)]})
    assert credit_windows(w3, ep, 24, offset_hours=0)[0] == 0 and credit_windows(w3, ep, 24, offset_hours=1)[0] == -1


def test_detection_and_earliest_lead():
    w = _windows()
    ep = pd.DataFrame({"episode": [0], "machine_id": [1], "failure_time": [T0]})
    flagged = np.zeros(len(w), bool)
    lead = (T0 - pd.to_datetime(w["target_end"])) / pd.Timedelta(hours=1)
    flagged[np.where((lead > 0) & (lead <= 24))[0][[0, 2]]] = True      # two warnings in the horizon
    flagged[np.where(lead > 24)[0][0]] = True                             # one too early: must not count
    res = detect_episodes(w, ep, flagged, 24).iloc[0]
    in_h = lead[(lead > 0) & (lead <= 24)]
    assert res.detected and res.eligible and res.flagged_windows == 2 and res.lead_hours == in_h.iloc[0]


def test_ineligible_when_horizon_not_observed():
    w = _windows(n=12, first_end=pd.Timestamp("2015-11-09 17:00"))      # 17:00, 23:00, 05:00: 3 of 4 slots
    ep = pd.DataFrame({"episode": [0], "machine_id": [1], "failure_time": [T0]})
    res = detect_episodes(w, ep, np.zeros(len(w), bool), 24).iloc[0]
    assert not res.eligible and res.windows_in_horizon == 3


def test_one_window_credits_one_episode():
    w = _windows(n=30)
    ep = pd.DataFrame({"episode": [0, 1], "machine_id": [1, 1], "failure_time": [T0, T0 + pd.Timedelta(hours=30)]})
    credit = credit_windows(w, ep, horizon_hours=48)
    lead0 = (T0 - pd.to_datetime(w["target_end"])) / pd.Timedelta(hours=1)
    # windows before T0 within 48 h go to episode 0 even though episode 1 is also within 48 h of some of them
    assert (credit[(lead0 > 0) & (lead0 <= 48)] == 0).all()
    assert set(np.unique(credit)) <= {-1, 0, 1}


def test_alarm_incidents_cooldown():
    w = _windows(n=12)
    flagged = np.zeros(12, bool)
    flagged[[0, 1, 4, 9]] = True       # 0,1 adjacent; 4 is 18 h after 1; 9 is 30 h after 4
    inc = alarm_incidents(w, flagged, cooldown_hours=24)
    assert inc[0] == inc[1] == inc[4] and inc[9] != inc[4] and (inc[~flagged] == -1).all()
    assert len(set(inc[flagged])) == 2


def test_alarm_windows_and_incidents_are_different_units():
    w = _windows(n=8)
    flagged = np.ones(8, bool)
    assert flagged.sum() == 8 and len(set(alarm_incidents(w, flagged, 24)[flagged])) == 1


# --- statistics ------------------------------------------------------------------------------------------------------

def test_bootstrap_resamples_whole_machines():
    machine_index = np.repeat(np.arange(4), [1, 5, 2, 7])               # unequal numbers of windows per machine
    values = torch.arange(len(machine_index), dtype=torch.float64)
    w = bootstrap_weights(4, 50, seed=1)
    assert torch.allclose(w.sum(1), torch.full((50,), 4.0, dtype=torch.float64))
    sums = machine_sums(values, machine_index, 4)
    counts = machine_sums(torch.ones(len(values)), machine_index, 4)
    # recompute replicate 0 by explicitly duplicating machines with all their windows
    expanded = np.concatenate([np.where(machine_index == m)[0] for m in range(4) for _ in range(int(w[0, m]))])
    assert np.isclose(weighted_ratio(sums, counts, w)[0].item(), values[expanded].mean().item())


def test_relative_degradation_uses_own_clean_mse():
    clean = {"mse": torch.tensor([1.0, 3.0]), "surprise": torch.tensor([1.0, 1.0])}
    corr = {"mse": torch.tensor([2.0, 4.0]), "surprise": torch.tensor([1.0, 1.0])}
    m = paired_metrics(clean, corr, torch.tensor([True, True]))
    assert np.isclose(m["relative_degradation"], 6 / 4 - 1)            # ratio of means, not mean of ratios


def test_threshold_uses_only_given_calibration_scores():
    val = torch.linspace(0, 1, 101)
    assert np.isclose(alert_threshold(val, 95), 0.95)


# --- protocol freezing -----------------------------------------------------------------------------------------------

def test_freeze_protocol_detects_changes_and_guards_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg, man, out = tmp / "s5.yaml", tmp / "manifest.json", tmp / "scores.csv.gz"
        cfg.write_text("a: 1\n")
        out.write_text("x")
        try:
            freeze_protocol(cfg, man, guard_paths=[out])
            raise AssertionError("froze although outputs existed")
        except RuntimeError:
            pass
        out.unlink()
        _, created = freeze_protocol(cfg, man, guard_paths=[out])
        assert created and json.loads(man.read_text())["sha256"]
        _, created = freeze_protocol(cfg, man, guard_paths=[out])
        assert not created
        cfg.write_text("a: 2\n")
        try:
            freeze_protocol(cfg, man)
            raise AssertionError("changed protocol was not detected")
        except RuntimeError:
            pass


# --- data and checkpoint level ---------------------------------------------------------------------------------------

def _configs():
    with (ROOT / "configs" / "stage4.yaml").open(encoding="utf-8") as f:
        s4 = yaml.safe_load(f)
    with (ROOT / "configs" / "stage5.yaml").open(encoding="utf-8") as f:
        s5 = yaml.safe_load(f)
    return s4, s5


_CACHE = {}


def _test_partition():
    if "test" not in _CACHE:
        md = pd.read_csv(ROOT / "data" / "processed" / "window_metadata.csv", parse_dates=TIME_COLUMNS)
        _, s5 = _configs()
        part = s5["data"]["evaluation_partition"]
        _CACHE["test"] = (load_azure_dataset(ROOT / "data" / "processed", part, md),
                          md[md["partition"] == part].reset_index(drop=True))
    return _CACHE["test"]


def test_checkpoints_load_and_evaluation_changes_nothing():
    s4, s5 = _configs()
    keys = [(k, s) for k in s5["models"]["keys"] for s in s5["models"]["seeds"]]
    paths = {ROOT / s5["models"]["checkpoint"].format(key=k.lower(), seed=s) for k, s in keys}
    assert len(keys) == 9 and len(paths) == 9
    x = {"context": torch.randn(16, 4, 6, 4), "context_mask": torch.ones(16, 4, 6, 4),
         "target": torch.randn(16, 6, 4), "target_mask": torch.ones(16, 6, 4)}
    for (k, s), exp_epoch in zip(keys, [e for k in s5["models"]["keys"] for e in s5["models"]["expected_selected_epochs"][k]]):
        model, ck = load_checkpoint(ROOT / s5["models"]["checkpoint"].format(key=k.lower(), seed=s))
        assert ck["seed"] == s and ck["epoch"] == exp_epoch
        assert sum(p.numel() for p in model.parameters()) == s5["models"]["expected_parameters"]
        assert ck.get("input_mask", "ones") == s5["models"]["input_mask"][k] == s4["models"][k]["input_mask"]
        before = state_fingerprint(model)
        evaluate_only(model, x)
        assert state_fingerprint(model) == before and not model.training


def test_clean_paired_policy_and_target_preservation_on_test():
    ds, md = _test_partition()
    s4, s5 = _configs()
    tl = WindowTimeline(md)
    original_ctx, original_tgt = ds.context.clone(), ds.target.clone()
    for name in s5["corruption"]["scenarios"]:
        spec = s4["evaluation"]["scenarios"][name]
        avail = tl.window_masks(scenario_schedule(spec, tl, 4, make_generator(s5["corruption"]["scenario_seed"], name)))
        again = tl.window_masks(scenario_schedule(spec, tl, 4, make_generator(s5["corruption"]["scenario_seed"], name)))
        assert torch.equal(avail, again)                                    # reproducible
        same = tl.machine_index[1:] == tl.machine_index[:-1]
        assert torch.equal(avail[1:][same][:, :-1], avail[:-1][same][:, 1:])  # overlapping windows agree
        a = policy_inputs(ds.context, avail, ds.target, "ones")
        c = policy_inputs(ds.context, avail, ds.target, "mask")
        assert torch.equal(a["context"], c["context"])                        # identical values for A/B/C
        assert (a["context_mask"] == 1).all() and torch.equal(c["context_mask"], avail)
        assert torch.equal(a["target"], original_tgt) and (a["target_mask"] == 1).all()
        if name == "clean":
            assert torch.equal(a["context"], original_ctx) and (avail == 1).all()
    assert torch.equal(ds.context, original_ctx) and torch.equal(ds.target, original_tgt)


def test_test_windows_are_isolated_from_training_inputs():
    ds, md = _test_partition()
    full = pd.read_csv(ROOT / "data" / "processed" / "window_metadata.csv", parse_dates=TIME_COLUMNS)
    others = full[full["partition"] != md["partition"].iloc[0]]
    assert set(md["sample_id"]).isdisjoint(set(others["sample_id"]))
    earlier = others[others["target_end"] < md["context_start"].min()]
    assert len(earlier) == len(others)                                     # every other window ends before test begins
    assert set(torch.load(ROOT / "data" / "processed" / "test_windows.pt").keys()) <= {
        "context", "context_mask", "target", "target_mask", "sample_id", "machine_id", "partition", "sensors"}


TESTS = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]


def run_all():
    results = []
    for test in TESTS:
        try:
            test()
            results.append((test.__name__, True, ""))
        except Exception as err:
            results.append((test.__name__, False, f"{type(err).__name__}: {err}"))
    return results


if __name__ == "__main__":
    outcome = run_all()
    for name, ok, msg in outcome:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}")
    sys.exit(0 if all(ok for _, ok, _ in outcome) else 1)
