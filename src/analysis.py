"""Compute head-direction and turning metrics from tracked body points."""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd

TAU = 2 * math.pi
REQUIRED = ("time_s", "nose_x", "nose_y", "neck_x", "neck_y",
            "likelihood_nose", "likelihood_neck", "condition")


def synthetic_landmarks(fps: int = 20, duration_s: float = 30.0,
                        turns_when_stimulated: float = 3.0) -> pd.DataFrame:
    """Generate two illustrative sessions with known rotational ground truth."""
    rng = np.random.default_rng(2026)
    t = np.arange(round(fps * duration_s)) / fps
    records = []
    for condition in ("baseline", "stimulation"):
        if condition == "baseline":
            angle = 0.35 * np.sin(0.29 * t) + 0.08 * np.sin(1.4 * t)
        else:
            # Three full positive revolutions over 30 seconds plus tiny head motion.
            angle = TAU * turns_when_stimulated * t / duration_s + 0.025 * np.sin(1.2 * t)
        cx = 250 + 12 * np.sin(0.11 * t)
        cy = 210 + 8 * np.cos(0.15 * t)
        radius = 17
        nx = cx + radius * np.cos(angle) + rng.normal(0, 0.18, len(t))
        ny = cy + radius * np.sin(angle) + rng.normal(0, 0.18, len(t))
        confidence = np.full(len(t), 0.99)
        confidence[np.arange(45, 49)] = 0.4   # low-confidence interval
        for i, ts in enumerate(t):
            records.append((condition, i, float(ts), float(nx[i]), float(ny[i]),
                            float(cx[i]), float(cy[i]), float(confidence[i]), 0.99))
    return pd.DataFrame(records, columns=[
        "condition", "frame", "time_s", "nose_x", "nose_y", "neck_x",
        "neck_y", "likelihood_nose", "likelihood_neck"
    ])


def _impute_points(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Interpolate excluded pose estimates, never use low-confidence points."""
    data = data.copy().sort_values("time_s").reset_index(drop=True)
    valid = (data.likelihood_nose >= threshold) & (data.likelihood_neck >= threshold)
    if valid.sum() < 3:
        raise ValueError("Not enough high-confidence landmarks for this session.")
    for col in ("nose_x", "nose_y", "neck_x", "neck_y"):
        data.loc[~valid, col] = np.nan
        data[col] = data[col].interpolate(limit_direction="both")
    return data


def head_orientation(data: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Compute circular head angles and continuous unwrapped orientation."""
    clean = _impute_points(data, threshold)
    nose = clean[["nose_x", "nose_y"]].to_numpy(dtype=float)
    neck = clean[["neck_x", "neck_y"]].to_numpy(dtype=float)
    vector = nose - neck
    phase = np.arctan2(vector[:, 1], vector[:, 0])
    delta = phase[1:] - phase[:-1]
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    continuous = np.concatenate((phase[:1], phase[0] + np.cumsum(delta)))
    clean["head_angle_deg"] = np.rad2deg(phase)
    clean["unwrapped_deg"] = np.rad2deg(continuous)
    diff = np.diff(clean["unwrapped_deg"].to_numpy(), prepend=clean["unwrapped_deg"].iloc[0])
    dt = np.diff(clean["time_s"].to_numpy(), prepend=clean["time_s"].iloc[0])
    clean["angular_speed_deg_s"] = np.divide(diff, dt, out=np.zeros_like(diff),
                                                where=dt > 0)
    return clean


def count_spins(angles_deg: np.ndarray, t: np.ndarray,
                min_speed_deg_s: float = 12.0,
                full_turn_deg: float = 330.0) -> dict:
    """Count directional near-complete turns from a smoothed angle trace."""
    angles = np.asarray(angles_deg, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(angles) != len(t) or len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError("Provide ordered angular samples with increasing times.")
    window = min(15, len(angles) if len(angles) % 2 else len(angles) - 1)
    smooth = (pd.Series(angles).rolling(window, center=True, min_periods=1)
              .mean().to_numpy())
    increments = np.diff(smooth)
    speed = increments / np.diff(t)
    state, cumulative, ccw, cw = 0, 0.0, 0, 0
    for step, vel in zip(increments, speed):
        sign = 1 if vel >= min_speed_deg_s else -1 if vel <= -min_speed_deg_s else 0
        if sign != state:
            # Reset the accumulated rotation when the turning direction changes.
            cumulative, state = 0.0, sign
        if sign:
            cumulative += abs(step)
            while cumulative >= full_turn_deg:
                if sign > 0:
                    ccw += 1
                else:
                    cw += 1
                cumulative -= full_turn_deg
    return {"counterclockwise_spins": ccw, "clockwise_spins": cw,
            "total_spins": ccw + cw}


def analyse_sessions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not set(REQUIRED).issubset(df.columns):
        raise ValueError("Missing columns: " + ", ".join(sorted(set(REQUIRED) - set(df.columns))))
    traces, results = [], []
    for name, group in df.groupby("condition", sort=False):
        out = head_orientation(group)
        spins = count_spins(out.unwrapped_deg.to_numpy(), out.time_s.to_numpy())
        total_angular_travel = np.abs(np.diff(out.unwrapped_deg)).sum()
        results.append({"condition": name, "frames": len(out),
                        "angular_travel_deg": round(float(total_angular_travel), 2), **spins})
        out["condition"] = name
        traces.append(out)
    return pd.concat(traces, ignore_index=True), pd.DataFrame(results)


def save_plot(traces: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for name, data in traces.groupby("condition", sort=False):
        ax.plot(data.time_s, data.unwrapped_deg - data.unwrapped_deg.iloc[0], label=name)
    ax.set(xlabel="Time (s)", ylabel="Unwrapped head rotation (degrees)",
           title="Head-direction traces (synthetic example)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--csv", help="Input landmark CSV. Otherwise synthetic data is generated.")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(args.csv) if args.csv else synthetic_landmarks()
    if not args.csv:
        original.to_csv(output/"synthetic_pose_data.csv", index=False)
    traces, summary = analyse_sessions(original)
    traces.to_csv(output/"head_angle_trace.csv", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    save_plot(traces, output/"head_angle_demo.png")
    print(summary.to_string(index=False))
