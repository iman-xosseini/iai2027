"""Stage 5 failure-event analysis (exploratory): events, episodes, causal alarm matching, alarm incidents.

Timing. A window's surprise compares the prediction with the OBSERVED target block, so the score exists only
once that block has ended: its alarm time is target_end (+ an optional availability offset), never
context_end or target_start. A window can warn of a failure at time f only if 0 < f - alarm_time <= horizon.

Counting. Several failed components at one timestamp are one event; events of a machine closer than a fixed
gap form one episode. Every window is credited to at most one episode (the first that follows it within the
horizon), so one long alert period cannot be credited to several failures, and an episode counts once
however many of its windows alert. Episodes whose horizon is not fully observed are ineligible, not missed.
"""
import numpy as np
import pandas as pd


def failure_events(failures, partition=None):
    """One row per distinct (machine_id, datetime), with the failed components and the number of source rows."""
    df = failures if partition is None else failures[failures["partition"] == partition]
    df = df.assign(datetime=pd.to_datetime(df["datetime"]))
    events = df.groupby(["machine_id", "datetime"], as_index=False).agg(
        components=("component", lambda c: ",".join(sorted(map(str, c)))), rows=("component", "size"))
    return events.sort_values(["machine_id", "datetime"]).reset_index(drop=True)


def failure_episodes(events, gap_hours):
    """Merge events of one machine that follow the machine's previous event by less than `gap_hours`.

    The episode time is its first event (the earliest time a warning must precede).
    """
    ev = events.sort_values(["machine_id", "datetime"]).reset_index(drop=True)
    gap = ev.groupby("machine_id")["datetime"].diff()
    ev["episode"] = (gap.isna() | (gap >= pd.Timedelta(hours=gap_hours))).cumsum() - 1
    episodes = ev.groupby("episode").agg(machine_id=("machine_id", "first"), failure_time=("datetime", "first"),
                                         last_event=("datetime", "last"), events=("datetime", "size"),
                                         rows=("rows", "sum"), components=("components", lambda c: ";".join(c)))
    return episodes.reset_index()


def alarm_times(windows, offset_hours=0):
    """Alarm time of every window: target_end + offset (windows: DataFrame with a target_end column)."""
    return pd.to_datetime(windows["target_end"]) + pd.Timedelta(hours=offset_hours)


def credit_windows(windows, episodes, horizon_hours, offset_hours=0):
    """Episode each window can warn of: the first episode of its machine with 0 < f - alarm_time <= horizon.

    windows: DataFrame (machine_id, target_end), one row per window in dataset order.
    Returns an int array, the episode id per window, or -1.
    """
    w = pd.DataFrame({"machine_id": windows["machine_id"].to_numpy(),
                      "alarm": alarm_times(windows, offset_hours).to_numpy(), "row": np.arange(len(windows))})
    e = episodes[["machine_id", "failure_time", "episode"]].assign(failure_time=lambda d: pd.to_datetime(d["failure_time"]))
    e = e.astype({"machine_id": w["machine_id"].dtype})
    merged = pd.merge_asof(w.sort_values("alarm"), e.sort_values("failure_time"), left_on="alarm", right_on="failure_time",
                           by="machine_id", direction="forward", allow_exact_matches=False,
                           tolerance=pd.Timedelta(hours=horizon_hours))
    out = np.full(len(windows), -1, dtype=np.int64)
    hit = merged["episode"].notna().to_numpy()
    out[merged["row"].to_numpy()[hit]] = merged["episode"].to_numpy()[hit].astype(np.int64)
    return out


def episode_coverage(windows, episodes, horizon_hours, offset_hours=0, stride_hours=6):
    """Windows of the episode's machine with alarm time in [f - horizon, f), and eligibility (all slots present)."""
    alarm = alarm_times(windows, offset_hours).to_numpy()
    machine = windows["machine_id"].to_numpy()
    rows = []
    for ep in episodes.itertuples():
        f = np.datetime64(pd.Timestamp(ep.failure_time))
        n = int(((machine == ep.machine_id) & (alarm < f) & (alarm >= f - np.timedelta64(horizon_hours, "h"))).sum())
        rows.append({"episode": ep.episode, "windows_in_horizon": n, "eligible": n == horizon_hours // stride_hours})
    return pd.DataFrame(rows)


def detect_episodes(windows, episodes, flagged, horizon_hours, offset_hours=0, stride_hours=6):
    """Per episode: eligibility, detection and lead time (hours) of the earliest credited warning.

    flagged: bool array [N windows] (surprise above the fixed threshold).
    """
    credit = credit_windows(windows, episodes, horizon_hours, offset_hours)
    alarm = alarm_times(windows, offset_hours).to_numpy()
    flagged = np.asarray(flagged, dtype=bool)
    cov = episode_coverage(windows, episodes, horizon_hours, offset_hours, stride_hours).set_index("episode")
    rows = []
    for ep in episodes.itertuples():
        mine = (credit == ep.episode)
        warn = mine & flagged
        lead = ((np.datetime64(pd.Timestamp(ep.failure_time)) - alarm[warn].min()) / np.timedelta64(1, "h")
                if warn.any() else np.nan)
        rows.append({"episode": ep.episode, "machine_id": ep.machine_id, "failure_time": ep.failure_time,
                     "eligible": bool(cov.loc[ep.episode, "eligible"]),
                     "windows_in_horizon": int(cov.loc[ep.episode, "windows_in_horizon"]),
                     "credited_windows": int(mine.sum()), "flagged_windows": int(warn.sum()),
                     "detected": bool(warn.any()), "lead_hours": float(lead)})
    return pd.DataFrame(rows)


def alarm_incidents(windows, flagged, cooldown_hours, offset_hours=0):
    """Group flagged windows into distinct incidents per machine.

    A flagged window starts a new incident unless the same machine had a flagged window whose alarm time lies
    within the preceding `cooldown_hours`. Returns an int array (incident id per window, -1 if not flagged).
    """
    flagged = np.asarray(flagged, dtype=bool)
    df = pd.DataFrame({"machine_id": windows["machine_id"].to_numpy(), "alarm": alarm_times(windows, offset_hours).to_numpy(),
                       "row": np.arange(len(windows))})[flagged].sort_values(["machine_id", "alarm"])
    gap = df.groupby("machine_id")["alarm"].diff()
    new = gap.isna() | (gap > pd.Timedelta(hours=cooldown_hours))
    out = np.full(len(windows), -1, dtype=np.int64)
    out[df["row"].to_numpy()] = (new.cumsum() - 1).to_numpy()
    return out


def chance_recall(window_alert_rate, horizon_hours, stride_hours=6):
    """Recall of a random alarm process with the same per-window rate: 1 - (1 - q)^(windows per horizon)."""
    return 1 - (1 - window_alert_rate) ** (horizon_hours // stride_hours)
