from __future__ import annotations
import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import csv

#keep paths here to run tuner without having to use args all the fricking time
DEFAULT_LOGS=["/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/imuandemgdata.csv","/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/*.csv","/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/logs/*.csv","logs/*.csv",]
#i keep tuning outputs in one designated place
DEFAULT_OUT_DIR = "/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/models"
#keep csv organized
@dataclass
class TremorTrace:
    file_path: str
    phase: np.ndarray
    predicted_phase: np.ndarray
    frequency_hz: np.ndarray
    amplitude: np.ndarray
    dt: float
    env_phase_gain: float
    env_amplitude_gain: float
#wrap the angle errors to [-pi,pi] so the math doesn't have to jump when crossing 2pi
def wrap_angle_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
#use fft hilbert to get phase and amplitude from imu
def analytic_signal_math(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.complex64)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)
def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
def resolve_log_paths(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        if os.path.isfile(pattern):
            files.append(pattern)
        else:
            files.extend([p for p in glob.glob(pattern) if os.path.isfile(p)])
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no log files found for patterns: {patterns}")
    return files
#estimate dt from time but fallback is 0.05s
def safe_dt_from_df(df: pd.DataFrame, fallback: float = 0.05) -> float:
    if "time" not in df.columns:
        dt = max(dt, 0.01)
        return fallback
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return fallback
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return fallback
    dt = float(np.median(d))
    if not np.isfinite(dt) or dt <= 0:
        return fallback
    return dt
#choose gyro channels first cuz tremor frequency mostly appears there
def pick_imu_signal(df: pd.DataFrame) -> np.ndarray:
    for col in ["gz", "gy", "gx", "az", "ay", "ax"]:
        if col in df.columns:
            sig = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            return sig
    raise ValueError("no imu columns found. Need one of gx/gy/gz/ax/ay/az.")
#estimate simple environment gains so replay scoring is closer to real behavior
def estimate_env_gains(
    phase: np.ndarray,
    freq_hz: np.ndarray,
    amp: np.ndarray,
    u_logged: Optional[np.ndarray],
    dt: float,
    default_b: float,
) -> Tuple[float, float]:
    if u_logged is None or len(u_logged) < 3:
        return float(default_b), 0.0
    phase = np.asarray(phase, dtype=np.float64)
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    amp = np.asarray(amp, dtype=np.float64)
    u_logged = np.asarray(u_logged, dtype=np.float64)
    phase_unwrapped = np.unwrap(phase)
    dphi = np.diff(phase_unwrapped) - (2.0 * np.pi * freq_hz[:-1] * dt)
    damp = np.diff(amp)
    u = u_logged[:-1]
    n = min(len(u), len(dphi), len(damp))
    if n < 3:
        return float(default_b), 0.0
    u = u[:n]
    dphi = dphi[:n]
    damp = damp[:n]
    denom = float(np.dot(u, u) + 1e-9)
    if denom <= 1e-9:
        return float(default_b), 0.0

    b_env = float(np.dot(u, dphi) / denom)
    k_amp = float(np.dot(u, damp) / denom)
    if not np.isfinite(b_env):
        b_env = float(default_b)
    if not np.isfinite(k_amp):
        k_amp = 0.0
    return b_env, k_amp
#support both replay format and imu logs in one loader
def load_trace_from_csv(path: str, args: argparse.Namespace) -> Optional[TremorTrace]:
    df = pd.read_csv(path)
    if df.empty:
        return None
    #replay style format if they even exist
    replay_required = {"imu_phase_last", "imu_freq_hz", "imu_amp_mean"}
    if replay_required.issubset(set(df.columns)):
        dt = safe_dt_from_df(df, fallback=0.05)
        phase = pd.to_numeric(df["imu_phase_last"], errors="coerce").to_numpy(dtype=np.float64)
        phase = np.unwrap(np.nan_to_num(phase, nan=0.0))
        freq = pd.to_numeric(df["imu_freq_hz"], errors="coerce").to_numpy(dtype=np.float64)
        freq = np.nan_to_num(freq, nan=0.0)
        amp = pd.to_numeric(df["imu_amp_mean"], errors="coerce").to_numpy(dtype=np.float64)
        amp = np.clip(np.nan_to_num(amp, nan=0.0), 0.0, None)
        if "phase_pred" in df.columns:
            phase_pred = pd.to_numeric(df["phase_pred"], errors="coerce").to_numpy(dtype=np.float64)
            phase_pred = np.unwrap(np.nan_to_num(phase_pred, nan=0.0))
        else:
            phase_pred = phase + (2.0 * np.pi * freq * float(args.phase_horizon_s))
        u_logged: Optional[np.ndarray] = None
        if "u_final" in df.columns:
            u_logged = pd.to_numeric(df["u_final"], errors="coerce").to_numpy(dtype=np.float64)
            u_logged = np.nan_to_num(u_logged, nan=0.0)
        elif "u_mpc" in df.columns:
            u_logged = pd.to_numeric(df["u_mpc"], errors="coerce").to_numpy(dtype=np.float64)
            u_logged = np.nan_to_num(u_logged, nan=0.0)
        b_env, k_amp_env = estimate_env_gains(
            phase=phase,
            freq_hz=freq,
            amp=amp,
            u_logged=u_logged,
            dt=dt,
            default_b=float(args.B_init),
        )

        n = min(len(phase), len(phase_pred), len(freq), len(amp))
        if n < 5:
            return None
        return TremorTrace(
            file_path=path,
            phase=phase[:n],
            predicted_phase=phase_pred[:n],
            frequency_hz=freq[:n],
            amplitude=amp[:n],
            dt=dt,
            env_phase_gain=b_env,
            env_amplitude_gain=k_amp_env,
        )
    #imu format
    dt = safe_dt_from_df(df, fallback=0.05)
    signal = pick_imu_signal(df)
    if len(signal) < 5:
        return None
    centered = signal - float(np.mean(signal))
    analytic = analytic_signal_math(centered)
    phase = np.unwrap(np.angle(analytic))
    amp = np.abs(analytic)
    freq = np.diff(phase, prepend=phase[0]) / (2.0 * np.pi * dt)
    phase_pred = phase + (2.0 * np.pi * freq * float(args.phase_horizon_s))

    #in raw mode start with defaults for environment gains.
    b_env = float(args.B_init)
    k_amp_env = 0.0
    return TremorTrace(
        file_path=path,
        phase=phase,
        predicted_phase=phase_pred,
        frequency_hz=freq,
        amplitude=amp,
        dt=dt,
        env_phase_gain=b_env,
        env_amplitude_gain=k_amp_env,
    )
#control brain
class TremorController:
    def __init__(self, settings: Dict[str, Any]):
        raw_dt = float(settings.get("dt", 0.05))
        self.dt = max(raw_dt, 0.01)
        self.gain_b = float(settings.get("B", -0.05))
        self.p_u = float(settings.get("lambda_u", 0.01))
        self.p_du = float(settings.get("lambda_du", 0.01))
        self.horizon_s = float(settings.get("horizon", 0.8))
        self.u_min = float(settings.get("u_min", -1.0))
        self.u_max = float(settings.get("u_max", 1.0))
        self.iters = int(settings.get("iters", 40))
    def compute_optimal_control(self, cur_phi: float, omega: float, target_phi: float) -> float:
        N = max(2, int(round(self.horizon_s / max(self.dt, 1e-6))))
        N = min(N, 200)
        A = np.tril(np.ones((N, N), dtype=np.float64))
        drift = cur_phi + np.arange(1, N + 1, dtype=np.float64) * (omega * self.dt)
        targets = np.full(N, target_phi, dtype=np.float64)
        H = (self.gain_b ** 2) * (A.T @ A) + self.p_u * np.eye(N, dtype=np.float64)
        D = np.zeros((N - 1, N), dtype=np.float64)
        for i in range(N - 1):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0
        H += self.p_du * (D.T @ D)
        g = (self.gain_b * A).T @ (drift - targets)
        u = np.zeros(N, dtype=np.float64)
        eigmax = float(np.linalg.eigvalsh(H).max()) if N > 1 else 1.0
        step = 1.0 / (eigmax + 1e-6)
        for _ in range(self.iters):
            u -= step * (H @ u + g)
            u = np.clip(u, self.u_min, self.u_max)
        return float(u[0])
    
#used cem to search for better mpc params from my replay data
class ParameterOptimizer:
    def __init__(self, traces: List[TremorTrace], args: argparse.Namespace):
        self.traces = traces
        self.args = args
    def evaluate(self, params: Dict[str, float]) -> float:
        total_cost = 0.0
        total_steps = 0
        for tr in self.traces:
            controller = TremorController(
                {
                    "dt": tr.dt,
                    "B": params["B"],
                    "lambda_u": params["lambda_u"],
                    "lambda_du": params["lambda_du"],
                    "horizon": params["horizon_seconds"],
                    "u_min": self.args.u_min,
                    "u_max": self.args.u_max,
                    "iters": self.args.mpc_iters,
                }
            )

            amp_scale = float(np.median(np.abs(tr.amplitude)) + 1e-6)
            prev_u = 0.0
            for t in range(len(tr.phase) - 1):
                phi = float(tr.phase[t])
                phi_pred = float(tr.predicted_phase[t])
                omega = float(tr.frequency_hz[t] * 2.0 * np.pi)
                amp = float(tr.amplitude[t])
                if not (np.isfinite(phi) and np.isfinite(phi_pred) and np.isfinite(omega) and np.isfinite(amp)):
                    continue

                # target anti phase (-180 degrees or -pi) to cancel out tremor
                target = phi + float(self.args.phase_offset_rad)
                u = controller.compute_optimal_control(phi_pred, omega, target)
                if not np.isfinite(u):
                    return 1e30

                # simulate one step using estimated tremor data
                phi_next = phi + omega * tr.dt + tr.env_phase_gain * u
                amp_next = max(0.0, amp + tr.env_amplitude_gain * u)
                target_next = float(tr.phase[t + 1] + self.args.phase_offset_rad)
                phase_err = float(wrap_angle_to_pi(phi_next - target_next))
                cost = (
                    self.args.w_phase * (phase_err ** 2)
                    + self.args.w_amp * ((amp_next / amp_scale) ** 2)
                    + self.args.w_u * (u ** 2)
                    + self.args.w_du * ((u - prev_u) ** 2)
                )
                total_cost += float(cost)
                total_steps += 1
                prev_u = u

        if total_steps == 0:
            return 1e30
        return float(total_cost / total_steps)

    def run_tuning(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, np.ndarray]:
        if self.args.B_min > self.args.B_max:
            raise ValueError("B-min must be <= B-max")
        if self.args.horizon_min > self.args.horizon_max:
            raise ValueError("horizon-min must be <= horizon-max")
        if self.args.lambda_u_min <= 0 or self.args.lambda_u_max <= 0:
            raise ValueError("lambda-u bounds must be > 0")
        if self.args.lambda_du_min <= 0 or self.args.lambda_du_max <= 0:
            raise ValueError("lambda-du bounds must be > 0")
        if self.args.lambda_u_init <= 0 or self.args.lambda_du_init <= 0:
            raise ValueError("lambda init values must be > 0")
        if self.args.pop_size < 2:
            raise ValueError("pop-size must be >= 2")
        if not (0.0 < self.args.elite_frac <= 1.0):
            raise ValueError("elite-frac must be in (0, 1]")
        rng = np.random.default_rng(self.args.seed)

        #tune lambda
        mean = np.array(
            [
                float(self.args.B_init),
                math.log10(float(self.args.lambda_u_init)),
                math.log10(float(self.args.lambda_du_init)),
                float(self.args.horizon_init),
            ],
            dtype=np.float64,
        )
        lows = np.array(
            [
                float(self.args.B_min),
                math.log10(float(self.args.lambda_u_min)),
                math.log10(float(self.args.lambda_du_min)),
                float(self.args.horizon_min),
            ],
            dtype=np.float64,
        )
        highs = np.array(
            [
                float(self.args.B_max),
                math.log10(float(self.args.lambda_u_max)),
                math.log10(float(self.args.lambda_du_max)),
                float(self.args.horizon_max),
            ],
            dtype=np.float64,
        )
        mean = np.clip(mean, lows, highs)
        span = np.maximum(highs - lows, 1e-9)
        std = span / 5.0
        std_floor = span / 100.0
        pop_size = int(self.args.pop_size)
        elite_count = max(2, int(pop_size * float(self.args.elite_frac)))
        elite_count = min(elite_count, pop_size)
        history: List[Dict[str, Any]] = []
        best = {"best_score": float("inf"), "best_params": {}}
        for gen in range(1, int(self.args.iters) + 1):
            samples = rng.normal(loc=mean, scale=std, size=(pop_size, 4))
            samples = np.clip(samples, lows, highs)
            scored: List[Tuple[float, np.ndarray, Dict[str, float]]] = []
            for s in samples:
                params = {
                    "B": float(s[0]),
                    "lambda_u": float(10.0 ** s[1]),
                    "lambda_du": float(10.0 ** s[2]),
                    "horizon_seconds": float(s[3]),
                }
                score = self.evaluate(params)
                scored.append((float(score), s, params))
            scored.sort(key=lambda x: x[0])
            elites = np.array([row[1] for row in scored[:elite_count]], dtype=np.float64)
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), std_floor)
            iter_best_score, _, iter_best_params = scored[0]
            mean_score = float(np.mean([x[0] for x in scored]))
            if iter_best_score < best["best_score"]:
                best = {
                    "best_score": float(iter_best_score),
                    "best_params": dict(iter_best_params),
                }

            #save each gen
            history.append(
                {
                    "iter": gen,
                    "best_score": float(iter_best_score),
                    "mean_score": mean_score,
                    "best_params": dict(iter_best_params),
                }
            )
            print(
                f"[iter {gen:02d}] "
                f"best_score={iter_best_score:.6f} "
                f"B={iter_best_params['B']:.5f} "
                f"lambda_u={iter_best_params['lambda_u']:.6f} "
                f"lambda_du={iter_best_params['lambda_du']:.6f} "
                f"h={iter_best_params['horizon_seconds']:.3f}"
            )
        return best, history, mean, std

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune MPC parameters from IMU/EMG CSV logs")
    parser.add_argument("--logs", nargs="+", default=DEFAULT_LOGS)
    parser.add_argument("--best-out", default=f"{DEFAULT_OUT_DIR}/mpc_cem_best_params.json")
    parser.add_argument("--history-out", default=f"{DEFAULT_OUT_DIR}/mpc_cem_tune_history.json")
    parser.add_argument("--mean-out", default=f"{DEFAULT_OUT_DIR}/mpc_cem_tune_mean.npy")
    parser.add_argument("--std-out", default=f"{DEFAULT_OUT_DIR}/mpc_cem_tune_std.npy")
    parser.add_argument("--tuned-config-out", default=f"{DEFAULT_OUT_DIR}/tuned_config.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=24)
    parser.add_argument("--elite-frac", type=float, default=0.2)
    parser.add_argument("--w-phase", type=float, default=1.0)
    parser.add_argument("--w-amp", type=float, default=0.4)
    parser.add_argument("--w-u", type=float, default=0.1)
    parser.add_argument("--w-du", type=float, default=0.1)
    parser.add_argument("--B-init", type=float, default=-0.05)
    parser.add_argument("--B-min", type=float, default=-0.6)
    parser.add_argument("--B-max", type=float, default=0.0)
    parser.add_argument("--lambda-u-init", type=float, default=0.01)
    parser.add_argument("--lambda-u-min", type=float, default=1e-5)
    parser.add_argument("--lambda-u-max", type=float, default=0.5)
    parser.add_argument("--lambda-du-init", type=float, default=0.01)
    parser.add_argument("--lambda-du-min", type=float, default=1e-5)
    parser.add_argument("--lambda-du-max", type=float, default=0.5)
    parser.add_argument("--horizon-init", type=float, default=0.8)
    parser.add_argument("--horizon-min", type=float, default=0.2)
    parser.add_argument("--horizon-max", type=float, default=1.5)
    parser.add_argument("--phase-horizon-s", type=float, default=0.2)
    parser.add_argument("--phase-offset-rad", type=float, default=math.pi)
    parser.add_argument("--u-min", type=float, default=-1.0)
    parser.add_argument("--u-max", type=float, default=1.0)
    parser.add_argument("--mpc-iters", type=int, default=120)
    return parser.parse_args()


def write_outputs(
    best: Dict[str, Any],
    history: List[Dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    traces: List[TremorTrace],
    args: argparse.Namespace,
) -> None:
    for path in [args.best_out, args.history_out, args.mean_out, args.std_out, args.tuned_config_out]:
        ensure_parent(path)
    with open(args.best_out, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    with open(args.history_out, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    np.save(args.mean_out, mean)
    np.save(args.std_out, std)

    #save tune configs
    control_dt = float(np.median([tr.dt for tr in traces])) if traces else 0.05
    tuned_config = {
        "mpc": {
            "B": float(best["best_params"]["B"]),
            "lambda_u": float(best["best_params"]["lambda_u"]),
            "lambda_du": float(best["best_params"]["lambda_du"]),
            "horizon_seconds": float(best["best_params"]["horizon_seconds"]),
            "u_min": float(args.u_min),
            "u_max": float(args.u_max),
            "iters": int(args.mpc_iters),
        },
        "control": {
            "interval_s": control_dt,
            "phase_offset_rad": float(args.phase_offset_rad),
        },
        "prediction": {
            "phase_horizon_s": float(args.phase_horizon_s),
        },
    }
    with open(args.tuned_config_out, "w", encoding="utf-8") as f:
        json.dump(tuned_config, f, indent=2)


def main() -> None:
    args = parse_args()
    log_files = resolve_log_paths(args.logs)
    traces: List[TremorTrace] = []
    for path in log_files:
        try:
            tr = load_trace_from_csv(path, args)
            if tr is None:
                print(f"[skip] {path} (not enough usable rows)")
                continue
            traces.append(tr)
            print(
                f"[load] {os.path.basename(path)} "
                f"n={len(tr.phase)} dt={tr.dt:.4f} "
                f"b_env={tr.env_phase_gain:.6f} k_amp={tr.env_amplitude_gain:.6f}"
            )
        except Exception as e:
            print(f"[skip] {path} ({e})")
    if not traces:
        raise ValueError("no usable traces loaded.")
    optimizer = ParameterOptimizer(traces, args)
    best, history, mean, std = optimizer.run_tuning()
    write_outputs(best, history, mean, std, traces, args)
    print("\nDone tuning.")
    print(f"best score: {best['best_score']:.6f}")
    print(f"best params: {best['best_params']}")
    print(f"best params file: {args.best_out}")
    print(f"history file: {args.history_out}")
    print(f"mean/std files: {args.mean_out}, {args.std_out}")
    print(f"tuned config file: {args.tuned_config_out}")

if __name__ == "__main__":
    main()
