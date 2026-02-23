from __future__ import annotations
import argparse
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

#absolute paths
DEFAULT_CSV_PATH = "/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/imuandemgdata.csv"
DEFAULT_OUTPUT_DIR = "/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/models"

#glue
def set_seed(seed: int) -> None:
    #lock randomness so experiments are repeatable
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def pick_device(device_name: str) -> torch.device:
    #using GPU speeds up NN training a lot so just try auto-detect
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
def wrap_to_pi(x: np.ndarray | float) -> np.ndarray | float:
    #keep angles in -pi, pi so errors don't jump when phase wraps
    return (x + np.pi) % (2.0 * np.pi) - np.pi
def analytic_signal_fft(x: np.ndarray) -> np.ndarray:
    #hilbert-ish via FFT to get analytic signal (amplitude + phase)
    #amplitude = abs(analytic), phase = angle(analytic)
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.complex64)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        # even-length handling (nyquist term)
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        # odd-length handling
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)
def safe_dt_from_time(time_values: Optional[np.ndarray], fallback: float = 0.01) -> float:
    #try to estimate dt from the time column fallback to realistic default if weird
    if time_values is None or len(time_values) < 2:
        return fallback
    t = np.asarray(time_values, dtype=np.float64)
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
def parse_action_levels(s: str) -> np.ndarray:
    #parse 0.0,0.25,0.5,1.0 into a sorted numpy array need >=2 levels
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("Need at least 2 action levels, e.g. '0.0,0.5,1.0'")
    vals = sorted(set(vals))
    return np.asarray(vals, dtype=np.float32)
def ensure_dir(path: str) -> None:
    #create a path folder if it doesnt exist
    os.makedirs(path, exist_ok=True)

#data container for single trace
@dataclass
class TraceData:
    signal_col: str
    dt: float
    stride: int
    phase: np.ndarray
    amp: np.ndarray
    freq_hz: np.ndarray
#load and preprocess the imu and emg into phase amplitude and frequency
def load_trace_data(
    csv_path: str,
    signal_col: str,
    dt_override: float,
    target_dt: float,
    stride: int,
) -> TraceData:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    # if requested column aint there just use useual imu names
    if signal_col not in df.columns:
        for c in ["gz", "gy", "gx", "az", "ay", "ax"]:
            if c in df.columns:
                signal_col = c
                break
        else:
            raise ValueError("No IMU signal column found. Need one of gx/gy/gz/ax/ay/az.")
    #read numeric signal and clean NaNs
    signal = pd.to_numeric(df[signal_col], errors="coerce").to_numpy(dtype=np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    #optional time column for dt estimation
    time_values: Optional[np.ndarray] = None
    if "time" in df.columns:
        time_values = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=np.float64)
        time_values = np.nan_to_num(time_values, nan=np.nan)
    #raw dt
    if dt_override > 0:
        raw_dt = dt_override
    else:
        raw_dt = safe_dt_from_time(time_values, fallback=0.01)
    #fallback if raw dt<target dt
    auto_stride = 1
    if target_dt > 0 and raw_dt > 0 and raw_dt < target_dt:
        auto_stride = max(1, int(round(target_dt / raw_dt)))
    total_stride = max(1, int(stride), int(auto_stride))
    #reduce data rate
    if total_stride > 1:
        signal = signal[::total_stride]
        if time_values is not None:
            time_values = time_values[::total_stride]
    #recalculate dt after subsampling
    if dt_override > 0:
        dt = dt_override * total_stride
    else:
        dt = safe_dt_from_time(time_values, fallback=raw_dt * total_stride)
    if dt <= 0 or not np.isfinite(dt):
        dt = 0.01
    if len(signal) < 12:
        raise ValueError(
            f"Not enough samples after stride={total_stride}. Need >=12, got {len(signal)}."
        )
    #remove annoying ah dc bias
    centered = signal - float(np.mean(signal))
    analytic = analytic_signal_fft(centered)
    phase = np.unwrap(np.angle(analytic))
    amp = np.abs(analytic)
    freq_hz = np.diff(phase, prepend=phase[0]) / (2.0 * np.pi * max(dt, 1e-6))
    return TraceData(
        signal_col=signal_col,
        dt=float(dt),
        stride=total_stride,
        phase=phase.astype(np.float64),
        amp=amp.astype(np.float64),
        freq_hz=freq_hz.astype(np.float64),
    )

#replay env to simulate effects of stimulation on tremor trace
class TremorReplayEnv:
    def __init__(
        self,
        phase: np.ndarray,
        amp: np.ndarray,
        freq_hz: np.ndarray,
        action_levels: np.ndarray,
        rollout_steps: int,
        phase_offset_rad: float,
        k_phase: float,
        k_amp: float,
        w_phase: float,
        w_amp: float,
        w_u: float,
        w_du: float,
        rng: np.random.Generator,
    ):
        #store recorded trace
        self.phase = np.asarray(phase, dtype=np.float64)
        self.amp = np.asarray(amp, dtype=np.float64)
        self.freq = np.asarray(freq_hz, dtype=np.float64)
        self.action_levels = np.asarray(action_levels, dtype=np.float32)
        #stim model params
        self.phase_offset_rad = float(phase_offset_rad)
        self.k_phase = float(k_phase)
        self.k_amp = float(k_amp)
        # reward weights *note to self tune these later if needed*
        self.w_phase = float(w_phase)
        self.w_amp = float(w_amp)
        self.w_u = float(w_u)
        self.w_du = float(w_du)
        self.rng = rng
        self.n = len(self.phase)
        if self.n < 12:
            raise ValueError("Need at least 12 points in trace for RL training.")
        #episode can't be longer than trace
        self.rollout_steps = max(8, min(int(rollout_steps), self.n - 2))
        #basic normalizers so neural net inputs are sane
        self.amp_scale = max(float(np.median(np.abs(self.amp))), 1e-6)
        self.freq_scale = max(float(np.percentile(np.abs(self.freq), 95)), 1e-6)
        #find index for "zero stimulation" action
        self.zero_action_idx = int(np.argmin(np.abs(self.action_levels)))
        self.state_dim = 7  #[sin(phi), cos(phi), amp_norm, freq_norm, prev_action, prev_err_norm, amp_delta_norm]
        #runtime variables
        self.t = 0
        self.start = 0
        self.steps = 0
        self.current_amp = 0.0
        self.prev_action = 0.0
        self.prev_phase_err = 0.0
    def reset(self, random_start: bool = True) -> np.ndarray:
        #help generalization by picking a random starting index
        if random_start and self.n > self.rollout_steps + 2:
            high = self.n - self.rollout_steps - 1
            self.start = int(self.rng.integers(0, high))
        else:
            self.start = 0
        self.t = self.start
        self.steps = 0
        self.current_amp = float(self.amp[self.t])
        self.prev_action = 0.0
        self.prev_phase_err = 0.0
        return self._state()
    def _state(self) -> np.ndarray:
        #build 7 element state vector
        phi = float(self.phase[self.t])
        freq = float(self.freq[self.t])
        amp_norm = float(np.clip(self.current_amp / self.amp_scale, 0.0, 5.0))
        freq_norm = float(np.clip(freq / self.freq_scale, -5.0, 5.0))
        prev_err_norm = float(np.clip(self.prev_phase_err / np.pi, -1.0, 1.0))
        if self.t > 0:
            amp_prev_baseline = float(self.amp[self.t - 1])
        else:
            amp_prev_baseline = float(self.amp[self.t])
        amp_delta_norm = float(
            np.clip((self.current_amp - amp_prev_baseline) / self.amp_scale, -2.0, 2.0)
        )
        return np.array(
            [
                math.sin(phi),          #sin(phase) nice continuous representation
                math.cos(phi),          #cos(phase)
                amp_norm,               #normalized tremor amplitude
                freq_norm,              #normalized frequency
                float(self.prev_action),#last action so agent knows its recent output
                prev_err_norm,          #previous phase error scaled to -1,1
                amp_delta_norm,         #recent amplitude change
            ],
            dtype=np.float32,
        )
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        #clamp action index just in case
        action_idx = int(np.clip(action_idx, 0, len(self.action_levels) - 1))
        action = float(self.action_levels[action_idx])
        #current phase and anti-phase target
        phi_now = float(self.phase[self.t])
        target = phi_now + self.phase_offset_rad
        #what would have happened without stimulation (recorded trace)
        base_phi_next = float(self.phase[self.t + 1])
        base_amp_next = float(self.amp[self.t + 1])
        #stim linearly nudges output
        sim_phi_next = base_phi_next + self.k_phase * action
        sim_amp_next = max(0.0, base_amp_next + (self.k_amp * self.amp_scale) * action)
        #compute error, amplitude norm, and action delta
        phase_err = float(wrap_to_pi(sim_phi_next - target))
        amp_norm = float(sim_amp_next / self.amp_scale)
        du = float(action - self.prev_action)
        #small phase error and amplitude change
        reward = -(
            self.w_phase * (phase_err ** 2)
            + self.w_amp * (amp_norm ** 2)
            + self.w_u * (action ** 2)
            + self.w_du * (du ** 2)
        )
        #step forward in time
        self.t += 1
        self.steps += 1
        self.current_amp = sim_amp_next
        self.prev_action = action
        self.prev_phase_err = phase_err
        done = bool(self.steps >= self.rollout_steps or self.t >= self.n - 2)
        next_state = self._state()
        info = {
            "amp_norm": amp_norm,
            "phase_err_rad": phase_err,
            "action": action,
        }
        return next_state, float(reward), done, info

#dqn tiny mlp buffer to map state to q value for each discrete action
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int):
        super().__init__()
        # 2 hidden layer mlp for this small state space
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
class ReplayBuffer:
    #circular buffer
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)
    def push(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: float) -> None:
        self.buffer.append((s, int(a), float(r), ns, float(done)))
    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        #random sample of transitions
        idx = rng.integers(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        s = np.stack([b[0] for b in batch]).astype(np.float32)
        a = np.array([b[1] for b in batch], dtype=np.int64)
        r = np.array([b[2] for b in batch], dtype=np.float32)
        ns = np.stack([b[3] for b in batch]).astype(np.float32)
        d = np.array([b[4] for b in batch], dtype=np.float32)
        return s, a, r, ns, d
    def __len__(self) -> int:
        return len(self.buffer)
def select_action(
    net: QNetwork,
    state: np.ndarray,
    epsilon: float,
    num_actions: int,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    #epsilon hella greedy
    if rng.random() < epsilon:
        return int(rng.integers(0, num_actions))

    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = net(s)
        return int(torch.argmax(q, dim=1).item())
def optimize_step(
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    #sample a batch and do one gradient update
    s, a, r, ns, d = replay.sample(batch_size, rng)
    s_t = torch.tensor(s, dtype=torch.float32, device=device)
    a_t = torch.tensor(a, dtype=torch.int64, device=device)
    r_t = torch.tensor(r, dtype=torch.float32, device=device)
    ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
    d_t = torch.tensor(d, dtype=torch.float32, device=device)
    #Q(s,a) for chosen actions
    q = policy_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
    #target = r + γ * (1 - done) * Q_target(s', argmax_a' Q_policy(s'))
    with torch.no_grad():
        next_actions = torch.argmax(policy_net(ns_t), dim=1, keepdim=True)
        next_q = target_net(ns_t).gather(1, next_actions).squeeze(1)
        target = r_t + gamma * (1.0 - d_t) * next_q
    loss = F.smooth_l1_loss(q, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    #clip grads so training don't explode
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


def run_rollout(
    env: TremorReplayEnv,
    action_fn,
    random_start: bool = False,
) -> Dict[str, float]:
    #run one episode to collect simple eval stats (reward/amp/phase/action)
    state = env.reset(random_start=random_start)
    total_reward = 0.0
    total_amp_norm = 0.0
    total_abs_phase_err = 0.0
    total_action = 0.0
    steps = 0
    done = False
    while not done:
        action_idx = int(action_fn(state))
        state, reward, done, info = env.step(action_idx)
        total_reward += reward
        total_amp_norm += float(info["amp_norm"])
        total_abs_phase_err += abs(float(info["phase_err_rad"]))
        total_action += abs(float(info["action"]))
        steps += 1
    if steps == 0:
        steps = 1
    return {
        "steps": float(steps),
        "reward_total": float(total_reward),
        "reward_per_step": float(total_reward / steps),
        "amp_norm_mean": float(total_amp_norm / steps),
        "phase_err_abs_mean": float(total_abs_phase_err / steps),
        "action_abs_mean": float(total_action / steps),
    }

#main training
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train offline DQN for tremor modulation policy")
    #data and outputs
    p.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH)
    p.add_argument("--signal-col", type=str, default="gz")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    #subsampling
    p.add_argument("--dt-override", type=float, default=0.0)
    p.add_argument("--target-dt", type=float, default=0.002, help="Auto-stride target dt (seconds)")
    p.add_argument("--stride", type=int, default=1)
    #action discretization and toy physics
    p.add_argument("--action-levels", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--phase-offset-rad", type=float, default=math.pi)
    p.add_argument("--k-phase", type=float, default=-0.04)
    p.add_argument("--k-amp", type=float, default=-0.15)
    #reward weights
    p.add_argument("--w-phase", type=float, default=0.25)
    p.add_argument("--w-amp", type=float, default=1.0)
    p.add_argument("--w-u", type=float, default=0.03)
    p.add_argument("--w-du", type=float, default=0.08)
    #rl training hyperparams
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--target-update-steps", type=int, default=250)
    #exploration schedule
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=float, default=120.0)
    #reproducibility
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    device = pick_device(args.device)
    #parse action levels from string to array
    action_levels = parse_action_levels(args.action_levels)
    # load trace and if needed auto stride
    trace = load_trace_data(
        csv_path=args.csv_path,
        signal_col=args.signal_col,
        dt_override=float(args.dt_override),
        target_dt=float(args.target_dt),
        stride=int(args.stride),
    )
    #build replay environment that simulates stim effects on recorded trace
    env = TremorReplayEnv(
        phase=trace.phase,
        amp=trace.amp,
        freq_hz=trace.freq_hz,
        action_levels=action_levels,
        rollout_steps=args.rollout_steps,
        phase_offset_rad=args.phase_offset_rad,
        k_phase=args.k_phase,
        k_amp=args.k_amp,
        w_phase=args.w_phase,
        w_amp=args.w_amp,
        w_u=args.w_u,
        w_du=args.w_du,
        rng=rng,
    )
    # policy and target nets
    policy_net = QNetwork(env.state_dim, len(action_levels), args.hidden_dim).to(device)
    target_net = QNetwork(env.state_dim, len(action_levels), args.hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size)
    #quick sanityprint so i can check inputs before a long run
    print(
        f"[data] file={args.csv_path} signal={trace.signal_col} n={len(trace.phase)} "
        f"dt={trace.dt:.6f}s stride={trace.stride}"
    )
    print(f"[rl] device={device} state_dim={env.state_dim} actions={action_levels.tolist()}")
    history_rows: List[Dict[str, float]] = []
    global_step = 0
    #main episodes loop
    for ep in range(1, args.episodes + 1):
        #simple epsilon decay over episodes
        eps = args.eps_end + (args.eps_start - args.eps_end) * math.exp(
            -(ep - 1) / max(args.eps_decay_episodes, 1e-6)
        )
        state = env.reset(random_start=True)
        done = False
        #episode accumulators for logging
        ep_reward = 0.0
        ep_amp = 0.0
        ep_abs_phase = 0.0
        ep_action = 0.0
        ep_steps = 0
        ep_losses: List[float] = []
        while not done:
            #pick action
            action_idx = select_action(
                net=policy_net,
                state=state,
                epsilon=eps,
                num_actions=len(action_levels),
                rng=rng,
                device=device,
            )
            #step env and store transition
            next_state, reward, done, info = env.step(action_idx)
            replay.push(state, action_idx, reward, next_state, float(done))
            state = next_state
            #logging and all
            ep_reward += reward
            ep_amp += float(info["amp_norm"])
            ep_abs_phase += abs(float(info["phase_err_rad"]))
            ep_action += abs(float(info["action"]))
            ep_steps += 1
            global_step += 1
            #can we train now?
            can_train = (
                len(replay) >= args.batch_size
                and global_step >= args.warmup_steps
                and (global_step % args.train_every == 0)
            )
            if can_train:
                loss = optimize_step(
                    policy_net=policy_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    replay=replay,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    rng=rng,
                    device=device,
                )
                ep_losses.append(loss)
            #periodically sync target network to stabilize learning
            if global_step % args.target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())
        #save a row of summary stats for this episode
        row = {
            "episode": float(ep),
            "epsilon": float(eps),
            "reward_total": float(ep_reward),
            "reward_per_step": float(ep_reward / max(ep_steps, 1)),
            "amp_norm_mean": float(ep_amp / max(ep_steps, 1)),
            "phase_err_abs_mean": float(ep_abs_phase / max(ep_steps, 1)),
            "action_abs_mean": float(ep_action / max(ep_steps, 1)),
            "loss_mean": float(np.mean(ep_losses)) if ep_losses else np.nan,
            "steps": float(ep_steps),
        }
        history_rows.append(row)
        #every now and then console prints
        if ep == 1 or ep == args.episodes or ep % max(1, args.episodes // 20) == 0:
            print(
                f"[ep {ep:03d}/{args.episodes}] "
                f"R={row['reward_total']:.3f} "
                f"R/step={row['reward_per_step']:.4f} "
                f"amp={row['amp_norm_mean']:.4f} "
                f"phase={row['phase_err_abs_mean']:.4f} "
                f"act={row['action_abs_mean']:.4f} "
                f"eps={row['epsilon']:.3f}"
            )
    #final sync
    target_net.load_state_dict(policy_net.state_dict())
    #compare doing nothing vs greedy policy
    zero_idx = int(env.zero_action_idx)
    def zero_policy(_: np.ndarray) -> int:
        return zero_idx
    def greedy_policy(s: np.ndarray) -> int:
        with torch.no_grad():
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            return int(torch.argmax(policy_net(st), dim=1).item())
    baseline_eval = run_rollout(env, zero_policy, random_start=False)
    trained_eval = run_rollout(env, greedy_policy, random_start=False)
    #save everything
    ensure_dir(args.output_dir)
    model_path = os.path.join(args.output_dir, "rl_dqn_model.pt")
    history_path = os.path.join(args.output_dir, "rl_training_history.csv")
    summary_path = os.path.join(args.output_dir, "rl_training_summary.json")
    meta_path = os.path.join(args.output_dir, "rl_inference_meta.json")
    norm_path = os.path.join(args.output_dir, "rl_state_norm.npz")
    checkpoint = {
        "model_state_dict": policy_net.state_dict(),
        "state_dim": env.state_dim,
        "num_actions": len(action_levels),
        "action_levels": action_levels.tolist(),
        "state_features": [
            "sin_phase",
            "cos_phase",
            "amp_norm",
            "freq_norm",
            "prev_action",
            "prev_phase_err_over_pi",
            "amp_delta_norm",
        ],
        "signal_col": trace.signal_col,
        "dt_seconds": trace.dt,
        "amp_scale": env.amp_scale,
        "freq_scale": env.freq_scale,
        "training_args": vars(args),
    }
    torch.save(checkpoint, model_path)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    best_row = max(history_rows, key=lambda x: x["reward_total"])
    summary = {
        "csv_path": args.csv_path,
        "signal_col": trace.signal_col,
        "num_samples_used": int(len(trace.phase)),
        "dt_seconds": float(trace.dt),
        "effective_stride": int(trace.stride),
        "device": str(device),
        "episodes": int(args.episodes),
        "best_episode": int(best_row["episode"]),
        "best_episode_reward_total": float(best_row["reward_total"]),
        "baseline_eval": baseline_eval,
        "trained_eval": trained_eval,
        "files": {
            "model_pt": model_path,
            "history_csv": history_path,
            "summary_json": summary_path,
            "meta_json": meta_path,
            "norm_npz": norm_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    inference_meta = {
        "model_path": model_path,
        "action_levels": action_levels.tolist(),
        "state_dim": int(env.state_dim),
        "state_features": checkpoint["state_features"],
        "signal_col": trace.signal_col,
        "dt_seconds": float(trace.dt),
        "normalization": {
            "amp_scale": float(env.amp_scale),
            "freq_scale": float(env.freq_scale),
        },
        "note": "Load model weights from rl_dqn_model.pt and build same QNetwork architecture.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(inference_meta, f, indent=2)
    np.savez(
        norm_path,
        amp_scale=np.array([env.amp_scale], dtype=np.float32),
        freq_scale=np.array([env.freq_scale], dtype=np.float32),
        dt_seconds=np.array([trace.dt], dtype=np.float32),
    )
    #final prints
    print("\nTraining complete.")
    print(f"Model:   {model_path}")
    print(f"History: {history_path}")
    print(f"Summary: {summary_path}")
    print(f"Meta:    {meta_path}")
    print(f"Norm:    {norm_path}")
    print("\nEvaluation (single rollout from start):")
    print(f"Baseline reward/step: {baseline_eval['reward_per_step']:.6f}")
    print(f"Trained  reward/step: {trained_eval['reward_per_step']:.6f}")

if __name__ == "__main__":
    main()
