from __future__ import annotations
import argparse
import sys
import time
import logging
import datetime
import serial
from serial.tools import list_ports
import csv
import json
import math
import os
import queue
import signal
import threading 
from typing import Any, Callable, Dict, List, Tuple, Optional
from collections import deque
import struct
import numpy as np
import tensorflow as tf
import yaml
from scipy.signal import butter, hilbert, sosfilt, sosfilt_zi, welch

LOG = logging.getLogger("realtime")
#my model training files
TRAINING_MODELS_DIR = "/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/models"
FRAME_HEADER = b"TSF"
FRAME_MAX_SAMPLES = 64
#configs
DEFAULTS: Dict[str, Any] = {
    "serial": {"port": "/dev/ttyACM0", "baud": 200000},
    "sampling": {"emg_fs": 860.0, "imu_fs": 200.0, "window_seconds": 0.5},
    "paths": {"models_dir": TRAINING_MODELS_DIR, "logs_dir": "./logs"},
    "model": {
        "keras_h5": "tremor_model.keras",
        "norm_np": "tremor_model_norm.npz",
        "input_type": "auto",
        "feature_order": [
            "emg_rms",
            "emg_env_mean",
            "emg_centroid_hz",
            "imu_amp_mean",
            "imu_phase_last",
            "imu_freq_hz",
            "imu_peak_hz",
            "emg_imu_corr",
            "plv_emg_imu",
            "tremor_power",
        ],
        "sequence_signals": ["emg_bp", "ax", "ay", "az", "gx", "gy", "gz"],
        "sequence_downsample": 1,
        "num_threads": 2,
        "output_map": ["tremor_prob", "phase_pred", "amp_pred"],
        "output_activation": {"tremor_prob": "sigmoid"},
    },
    "prediction": {
        "phase_horizon_s": 0.2,
        "amp_horizon_s": 0.2,
        "tremor_prob_thresh": 0.5,
        "amp_thresh": 0.15,
        "freq_range_hz": [3.0, 8.0],
    },
    "dsp": {
        "emg_band": [20.0, 250.0],
        "imu_band": [3.0, 12.0],
        "emg_env_lp_hz": 5.0,
        "axis_default": "gz",
        "axis_autoselect": True,
    },
    "control": {
        "mode": "hybrid", 
        "phase_offset_rad": math.pi,
        "k_phase": 0.8,
        "k_amp": 0.4,
        "interval_s": 0.05,
        "u_min": -1.0,
        "u_max": 1.0,
        "rl_blend": 0.35, #lower than mpc cuz it less predictable than mpc and can be risky
        "mpc_blend": 0.65,
    },
    "mpc": {
        "horizon_seconds": 0.8,
        "B": -0.05,
        "lambda_u": 0.01,
        "lambda_du": 0.01,
        "u_min": -1.0,
        "u_max": 1.0,
        "iters": 120,
    },
    "rl": {
        "enabled": True,
        "learning_rate": 0.015,
        "gamma": 0.98,
        "weight_decay": 1e-3,
        "action_scale": 0.8,
        "exploration_sigma": 0.05,
        "state_features": [
            "imu_amp_mean",
            "imu_freq_hz",
            "emg_rms",
            "phase_error",
            "tremor_prob",
            "dopamine",
        ],
        "save_path": "rl_state.json",
        "load_path": "rl_state.json",
    },
    "dopamine": {
        "baseline": 0.35,
        "tau_s": 2.0,
        "k_reward": 1.0,
        "k_decay": 0.5,
        "rpe_smoothing": 0.95,
        "clip": 2.0,
    },
    "metrics": {
        "latency_target_ms": 50.0,
        "energy_scale": 0.06,
        "amp_baseline_tau_s": 5.0,
        "reward_weights": {"amp": 1.0, "energy": 0.4, "latency": 0.2},
    },
    "safety": {
        "control_limit_per_second": 10,
        "emergency_stop_threshold": 0.8,
        "emergency_stop_duration": 1.0,
        "resume_threshold": 0.5,
        "resume_duration": 2.0,
        "watchdog_s": 0.2,
        "ack_timeout_s": 0.05,
        "ack_required": True,
    },
    "sensor_bounds": {
        "emg": [-5000.0, 5000.0],
        "gx": [-32768.0, 32767.0],
        "gy": [-32768.0, 32767.0],
        "gz": [-32768.0, 32767.0],
    },
    "stimulation": {
        "freq_hz": {"base": 120.0, "range": 40.0},
        "pulse_width_us": {"base": 200.0, "range": 150.0},
        "burst_ms": {"base": 150.0, "range": 100.0},
    },
    "logging": {
        "log_frequency_hz": 10,
        "max_log_file_size_mb": 50,
        "backup_count": 5,
    },
    "debug": {
        "enable_debug_mode": False,
        "debug_level": "INFO",
        "simulate_serial_data": False,
        "simulated_data_file": "simulated_data.csv",
        "simulate_real_time": True,
        "simulation_speed_factor": 1.0,
    },
    "output": {
        "serial_control": {
            "enabled": False,
            "port": "/dev/ttyACM0",
            "baud": 115200,
            "mode": "stim",
            "format": "STIM,{freq_hz:.2f},{pulse_width_us:.2f},{burst_ms:.2f}\n",
            "max_rate_hz": 100.0,
            "stim_limits": {
                "freq_hz": [1.0, 200.0],
                "pulse_width_us": [10.0, 500.0],
                "burst_ms": [1.0, 200.0],
            },
        }
    },
    "hardware": {
        "emg_adc_bits": 12,
        "emg_adc_offset": 2048,
        "emg_vref": 3.3,
        "gyro_scale_per_lsb": 0.01,
    },
}
def cfg_get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out
def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if not path:
        return cfg
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in [".yaml", ".yml"]:
            user_cfg = yaml.safe_load(f) or {}
        else:
            user_cfg = json.load(f) or {}
    return deep_merge(cfg, user_cfg)
def to_abs(base_dir: str, maybe_path: Optional[str]) -> Optional[str]:
    if not maybe_path:
        return None
    if os.path.isabs(maybe_path):
        return maybe_path
    return os.path.join(base_dir, maybe_path)
def load_json_if_works(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        LOG.warning("could not read JSON file %s: %s", path, e) #don't think this should be a warning, but idk what else to do if the file is corrupted. maybe we should back it up and start fresh?
        return None
def apply_training_models_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    #auto pull trained artifacts
    model_dir = cfg_get(cfg, "paths.models_dir", TRAINING_MODELS_DIR)
    cfg.setdefault("paths", {})["models_dir"] = model_dir
    #apply best mpc configs
    tuned_cfg_path = os.path.join(model_dir, "tuned_config.json")
    tuned_cfg = load_json_if_works(tuned_cfg_path)
    if tuned_cfg:
        #only import the tuned mpc params
        cfg.setdefault("mpc", {})
        cfg["mpc"] = deep_merge(cfg["mpc"], tuned_cfg.get("mpc", {}))
    #if mpc configs missing just apply mpc params
    best_mpc_path = os.path.join(model_dir, "mpc_cem_best_params.json")
    best_mpc = load_json_if_works(best_mpc_path)
    if best_mpc and "best_params" in best_mpc:
        best = best_mpc["best_params"]
        cfg.setdefault("mpc", {})
        for key in ["B", "lambda_u", "lambda_du", "horizon_seconds"]:
            if key in best:
                cfg["mpc"][key] = best[key]
    #model and norm paths
    model_path = to_abs(model_dir, cfg_get(cfg, "model.keras_h5", "tremor_model.keras"))
    norm_path = to_abs(model_dir, cfg_get(cfg, "model.norm_np", "tremor_model_norm.npz"))
    if model_path and not os.path.exists(model_path):
        LOG.warning("Model file not found: %s", model_path)
        model_path = None
    if norm_path and not os.path.exists(norm_path):
        LOG.warning("Norm file not found: %s", norm_path)
        norm_path = None
    cfg.setdefault("model", {})["keras_h5"] = model_path
    cfg.setdefault("model", {})["norm_np"] = norm_path
    #rl paths
    rl_load = to_abs(model_dir, cfg_get(cfg, "rl.load_path", "rl_state.json"))
    rl_save = to_abs(model_dir, cfg_get(cfg, "rl.save_path", "rl_state.json"))
    cfg.setdefault("rl", {})["load_path"] = rl_load if rl_load and os.path.exists(rl_load) else None
    cfg.setdefault("rl", {})["save_path"] = rl_save
    return cfg
#dynamic signal processing stuff
class StreamingSOSFilter:
    #iir filter
    def __init__(self, fs: float, cutoff: Tuple[float, float] | float, kind: str, order: int = 4):
        nyq = 0.5 * fs
        if isinstance(cutoff, tuple):
            wn = [cutoff[0] / nyq, cutoff[1] / nyq]
        else:
            wn = cutoff / nyq
        self.sos = butter(order, wn, btype=kind, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0
    def step(self, x: float) -> float:
        y, self.zi = sosfilt(self.sos, [x], zi=self.zi)
        return float(y[-1])
def analytic_signal(x: np.ndarray) -> np.ndarray:
    return hilbert(x)
def spectral_features(x: np.ndarray, fs: float) -> Tuple[float, float]:
    if len(x) < 8:
        return 0.0, 0.0
    x0 = x - np.mean(x)
    fft = np.fft.rfft(x0)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(x0), d=1.0 / fs)
    mag_sum = np.sum(mag) + 1e-12
    centroid = float(np.sum(freqs * mag) / mag_sum)
    peak = float(freqs[int(np.argmax(mag))])
    return centroid, peak
def band_power(x: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    if len(x) < 8:
        return 0.0
    f, pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    mask = (f >= band[0]) & (f <= band[1])
    return float(np.sum(pxx[mask]))
def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    if len(a) != len(b):
        b = np.interp(np.linspace(0, 1, len(a)), np.linspace(0, 1, len(b)), b)
    c = np.corrcoef(a, b)
    val = c[0, 1]
    return float(val) if np.isfinite(val) else 0.0
def phase_slope(phase: np.ndarray, fs: float) -> float:
    if len(phase) < 4:
        return 0.0
    t = np.arange(len(phase)) / fs
    p = np.unwrap(phase)
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, p, rcond=None)[0]
    return float(slope)
def plv(phase_a: np.ndarray, phase_b: np.ndarray) -> float:
    if len(phase_a) < 8 or len(phase_b) < 8:
        return 0.0
    if len(phase_a) != len(phase_b):
        phase_b = np.interp(np.linspace(0, 1, len(phase_a)), np.linspace(0, 1, len(phase_b)), phase_b)
    return float(np.abs(np.mean(np.exp(1j * (phase_a - phase_b)))))
#flow state (streaming)
class RingBuffer:
    def __init__(self, size: int):
        self.size = size
        self.data = np.zeros(size, dtype=np.float32)
        self.index = 0
        self.full = False
    def append(self, x: float) -> None:
        self.data[self.index] = x
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True
    def get(self) -> np.ndarray:
        if not self.full:
            return self.data[:self.index].copy()
        return np.concatenate((self.data[self.index:], self.data[:self.index]))
    def __len__(self) -> int:
        return self.size if self.full else self.index
class Decimator:
    def __init__(self, ratio: float):
        self.ratio = max(1.0, ratio)
        self.acc = 0.0
    def step(self) -> bool:
        self.acc += 1.0
        if self.acc >= self.ratio:
            self.acc -= self.ratio
            return True
        return False
class EWMA:
    def __init__(self, tau_s: float, dt: float, initial: float = 0.0):
        self.alpha = 1.0 - math.exp(-dt / max(1e-6, tau_s))
        self.value = initial
        self.ready = False
    def update(self, x: float) -> float:
        if not self.ready:
            self.value = x
            self.ready = True
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * x
        return self.value
#input (serial usb teensy)
class SerialFrameReader(threading.Thread):
    #after teensy sends packets this class unpacks them and queues them
    def __init__(
        self,
        port: str,
        baud: int,
        out_queue: "queue.Queue",
        stop_event: threading.Event,
        emg_fs: float,
        ack_event: Optional[threading.Event] = None,
        ack_time_ref: Optional[Dict[str, float]] = None,
    ):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.out_q = out_queue
        self.stop_event = stop_event
        self.dt_us = int(1e6 / emg_fs)
        self.ack_event = ack_event
        self.ack_time_ref = ack_time_ref
        self.send_lock = threading.Lock()
        self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
        LOG.info("Opened serial %s @ %d", self.port, self.baud)
    def run(self) -> None:
        buf = bytearray()
        while not self.stop_event.is_set():
            try:
                chunk = self.ser.read(4096)
                if not chunk:
                    continue
                buf.extend(chunk)
                while True:
                    #treat ack as text
                    if buf.startswith(b"ACK,"):
                        nl = buf.find(b"\n")
                        if nl == -1:
                            break
                        if self.ack_event is not None:
                            if self.ack_time_ref is not None:
                                self.ack_time_ref["mono"] = time.monotonic()
                            self.ack_event.set()
                        del buf[:nl + 1]
                        continue
                    i = buf.find(FRAME_HEADER)
                    if i < 0:
                        if len(buf) > 2:
                            del buf[:-2]
                        break
                    #drop trash
                    if i > 0:
                        del buf[:i]
                    #fixed header bytes
                    if len(buf) < 15:
                        break
                    n = buf[14]
                    if n <= 0 or n > FRAME_MAX_SAMPLES:
                        del buf[:1]
                        continue
                    need = 15 + n * 8
                    if len(buf) < need:
                        break
                    t0 = struct.unpack_from("<Q", buf, 6)[0]
                    payload_off = 15
                    for k in range(n):
                        emg16, gx16, gy16, gz16 = struct.unpack_from("<hhhh", buf, payload_off + k * 8)
                        t_us = t0 + k * self.dt_us
                        self.out_q.put((t_us, emg16, gx16, gy16, gz16))
                    del buf[:need]
            except Exception as e:
                if not self.stop_event.is_set():
                    LOG.warning("serial read error: %s", e)
                time.sleep(0.05)
    def send_raw(self, msg: str) -> None:
        with self.send_lock:
            try:
                self.ser.write(msg.encode("utf-8"))
            except Exception:
                pass
    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass
def load_simulated_csv(path: str) -> List[Tuple[float, float, float, float, float]]:
    rows: List[Tuple[float, float, float, float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("time_s") or row.get("time_us") or row.get("time")
            if t is None:
                continue
            t_val = float(t)
            if t_val > 1e6:
                t_val *= 1e-6
            emg = float(row.get("emg", row.get("emg_raw", 0.0)))
            gx = float(row.get("gx", 0.0))
            gy = float(row.get("gy", 0.0))
            gz = float(row.get("gz", 0.0))
            rows.append((t_val, emg, gx, gy, gz))
    return rows
class SimulatedReader(threading.Thread):
    def __init__(self,data: List[Tuple[float, float, float, float, float]],out_queue: "queue.Queue",stop_event: threading.Event,real_time: bool,speed: float):
        super().__init__(daemon=True)
        self.data = data
        self.out_q = out_queue
        self.stop_event = stop_event
        self.real_time = real_time
        self.speed = max(1e-3, speed)
    def run(self) -> None:
        if not self.data:
            LOG.error("no simulated data to replay")
            return
        t0 = self.data[0][0]
        wall0 = time.time()
        for t_s, emg, gx, gy, gz in self.data:
            if self.stop_event.is_set():
                break
            if self.real_time:
                target = wall0 + (t_s - t0) / self.speed
                while time.time() < target and not self.stop_event.is_set():
                    time.sleep(0.001)
            self.out_q.put((int(t_s * 1e6), emg, gx, gy, gz))
#output bridge
class ControlOutput:
    def __init__(self, cfg: Dict[str, Any]):
        self.enabled = bool(cfg_get(cfg, "output.serial_control.enabled", False))
        self.port = cfg_get(cfg, "output.serial_control.port", "/dev/ttyACM0")
        self.baud = int(cfg_get(cfg, "output.serial_control.baud", 115200))
        self.mode = cfg_get(cfg, "output.serial_control.mode", "stim")
        self.format = cfg_get(cfg,"output.serial_control.format","STIM,{freq_hz:.2f},{pulse_width_us:.2f},{burst_ms:.2f}\n")
        self.max_rate_hz = float(cfg_get(cfg, "output.serial_control.max_rate_hz", 100.0))
        self.min_interval = 1.0 / max(1.0, self.max_rate_hz)
        self.last_send_mono = 0.0
        self.ack_required = bool(cfg_get(cfg, "safety.ack_required", True))
        self.ack_timeout_s = float(cfg_get(cfg, "safety.ack_timeout_s", 0.05))
        self.stim_limits = cfg_get(cfg, "output.serial_control.stim_limits", {})
        self.stim_base = cfg_get(cfg, "stimulation", {})
        self.ack_event: Optional[threading.Event] = None
        self.ack_time_ref: Optional[Dict[str, float]] = None
        self.send_raw_fn: Optional[Callable[[str], None]] = None
        self.shared_serial = False
        self.ser = None
        if self.enabled:
            input_port = cfg_get(cfg, "serial.port", "/dev/ttyACM0")
            if self.port == input_port:
                self.shared_serial = True
                LOG.info("Control output sharing serial transport on %s", self.port)
            else:
                self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
                LOG.info("Control output serial opened %s @ %d", self.port, self.baud)
    def attach_ack(self, ack_event: threading.Event, ack_time_ref: Dict[str, float]) -> None:
        self.ack_event = ack_event
        self.ack_time_ref = ack_time_ref
    def attach_sender(self, send_raw_fn: Callable[[str], None]) -> None:
        self.send_raw_fn = send_raw_fn
    def _write(self, msg: str) -> bool:
        if self.shared_serial:
            if self.send_raw_fn is None:
                return False
            self.send_raw_fn(msg)
            return True
        if self.ser is None:
            return False
        try:
            self.ser.write(msg.encode("utf-8"))
            return True
        except Exception:
            return False
    def _clamp(self, name: str, value: float) -> float:
        limits = self.stim_limits.get(name, None)
        if not limits:
            return value
        return float(max(limits[0], min(limits[1], value)))
    def _map_stim(self, u: float) -> Dict[str, float]:
        #convert the normal control into stim params
        intensity = 0.5 + 0.5 * float(np.clip(u, -1.0, 1.0))
        freq_base = self.stim_base.get("freq_hz", {}).get("base", 120.0)
        freq_rng = self.stim_base.get("freq_hz", {}).get("range", 40.0)
        pw_base = self.stim_base.get("pulse_width_us", {}).get("base", 200.0)
        pw_rng = self.stim_base.get("pulse_width_us", {}).get("range", 150.0)
        burst_base = self.stim_base.get("burst_ms", {}).get("base", 150.0)
        burst_rng = self.stim_base.get("burst_ms", {}).get("range", 100.0)
        freq = self._clamp("freq_hz", freq_base + freq_rng * intensity)
        pulse_width = self._clamp("pulse_width_us", pw_base + pw_rng * intensity)
        burst = self._clamp("burst_ms", burst_base + burst_rng * intensity)
        return {"freq_hz": freq, "pulse_width_us": pulse_width, "burst_ms": burst}
    def send(self, u: float, phase: float, amp: float) -> bool:
        if not self.enabled:
            return False
        if self.shared_serial and self.send_raw_fn is None:
            return False
        if (not self.shared_serial) and self.ser is None:
            return False
        now = time.monotonic()
        if now - self.last_send_mono < self.min_interval:
            return False
        self.last_send_mono = now
        if self.ack_required and self.ack_event is not None:
            self.ack_event.clear()
        if self.mode == "stim":
            stim = self._map_stim(u)
            msg = self.format.format(freq_hz=stim["freq_hz"],pulse_width_us=stim["pulse_width_us"],burst_ms=stim["burst_ms"],u=u,phase=phase,amp=amp)
        else:
            msg = self.format.format(u=u, phase=phase, amp=amp)
        if not self._write(msg):
            return False
        if self.ack_required and self.ack_event is not None:
            if not self.ack_event.wait(self.ack_timeout_s):
                if not self._write(msg):
                    self.safe_disable()
                    return False
                if not self.ack_event.wait(self.ack_timeout_s):
                    self.safe_disable()
                    return False
        return True
    def safe_disable(self) -> None:
        if not self.enabled:
            return
        if self.shared_serial:
            if self.send_raw_fn is not None:
                self.send_raw_fn("STIM,0,0,0\n")
            return
        if self.ser is not None:
            try:
                self.ser.write(b"STIM,0,0,0\n")
            except Exception:
                pass
    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
#inference engine
class InferenceEngine:
    def __init__(self, model_path: Optional[str], norm_path: Optional[str], num_threads: int = 2):
        self.model_path = model_path
        self.model = None
        self.input_shapes: List[Tuple[Any, ...]] = []
        self.output_names: List[str] = []
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        self.norm_dict: Optional[Dict[str, Any]] = None
        try:
            tf.config.threading.set_intra_op_parallelism_threads(max(1, int(num_threads)))
            tf.config.threading.set_inter_op_parallelism_threads(max(1, int(num_threads)))
        except Exception:
            pass
        if norm_path and os.path.exists(norm_path):
            self._load_norm(norm_path)
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.input_shapes = [tuple(t.shape) for t in self.model.inputs]
            names = getattr(self.model, "output_names", None)
            if not names:
                names = [getattr(t, "name", f"output_{i}") for i, t in enumerate(getattr(self.model, "outputs", []))]
            if not names:
                names = ["output_0"]
            self.output_names = [str(n).split(":")[0] for n in names]
            LOG.info("Loaded model: %s", model_path)
        else:
            LOG.warning("No model loaded, using heuristic predictions.")
    def _load_norm(self, norm_path: str) -> None:
        try:
            obj = np.load(norm_path, allow_pickle=True)
            if isinstance(obj, np.lib.npyio.NpzFile):
                files = set(obj.files)
                if "mean" in files and "std" in files:
                    self.norm_mean = np.asarray(obj["mean"], dtype=np.float32)
                    self.norm_std = np.asarray(obj["std"], dtype=np.float32)
                    return
                if "arr_0" in files:
                    maybe = obj["arr_0"]
                    if isinstance(maybe, np.ndarray) and maybe.dtype == object and maybe.size == 1:
                        val = maybe.item()
                        if isinstance(val, dict):
                            self.norm_dict = val
                            return
            elif isinstance(obj, np.ndarray) and obj.shape[0] == 2:
                self.norm_mean = np.asarray(obj[0], dtype=np.float32)
                self.norm_std = np.asarray(obj[1], dtype=np.float32)
        except Exception as e:
            LOG.warning("Failed to load norm file: %s", e)
    def apply_norm(self, x: Any) -> Any:
        try:
            if self.norm_mean is not None and self.norm_std is not None:
                if isinstance(x, np.ndarray):
                    return (x - self.norm_mean) / (self.norm_std + 1e-8)
            if self.norm_dict is not None and isinstance(x, list) and len(x) >= 2:
                emg = np.asarray(x[0], dtype=np.float32)
                imu = np.asarray(x[1], dtype=np.float32)
                emg_mean = self.norm_dict.get("emg_mean", 0.0)
                emg_std = self.norm_dict.get("emg_std", 1.0)
                imu_mean = self.norm_dict.get("imu_mean", 0.0)
                imu_std = self.norm_dict.get("imu_std", 1.0)
                emg = (emg - emg_mean) / (emg_std + 1e-8)
                imu = (imu - imu_mean) / (imu_std + 1e-8)
                return [emg, imu]
        except Exception:
            pass
        return x
    def predict(self, x: Any) -> Any:
        if self.model is None:
            return None
        return self.model.predict(x, verbose=0)
#safety stuff
class SafetyGuard:
    def __init__(self, cfg: Dict[str, Any]):
        self.limit_per_sec = int(cfg_get(cfg, "safety.control_limit_per_second", 10))
        self.emergency_thresh = float(cfg_get(cfg, "safety.emergency_stop_threshold", 0.8))
        self.emergency_duration = float(cfg_get(cfg, "safety.emergency_stop_duration", 1.0))
        self.resume_thresh = float(cfg_get(cfg, "safety.resume_threshold", 0.5))
        self.resume_duration = float(cfg_get(cfg, "safety.resume_duration", 2.0))
        self.last_commands: deque = deque()
        self.emergency_start = None
        self.resume_start = None
        self.emergency_active = False
    def rate_limit(self, now: float) -> bool:
        while self.last_commands and now - self.last_commands[0] > 1.0:
            self.last_commands.popleft()
        if len(self.last_commands) >= self.limit_per_sec:
            return False
        self.last_commands.append(now)
        return True
    def apply(self, u: float, amp: float, now: float) -> Tuple[float, bool]:
        if amp >= self.emergency_thresh:
            if self.emergency_start is None:
                self.emergency_start = now
            if now - self.emergency_start >= self.emergency_duration:
                self.emergency_active = True
        else:
            self.emergency_start = None
        if self.emergency_active:
            if amp <= self.resume_thresh:
                if self.resume_start is None:
                    self.resume_start = now
                if now - self.resume_start >= self.resume_duration:
                    self.emergency_active = False
                    self.resume_start = None
            else:
                self.resume_start = None
            return 0.0, True

        if not self.rate_limit(now):
            return 0.0, False
        return u, False
class MPCControllerLite:
    #projected gradient descent
    def __init__(self, cfg: Dict[str, Any], dt: float):
        self.dt = dt
        self.B = float(cfg_get(cfg, "mpc.B", -0.05))
        self.lambda_u = float(cfg_get(cfg, "mpc.lambda_u", 0.01))
        self.lambda_du = float(cfg_get(cfg, "mpc.lambda_du", 0.01))
        self.u_min = float(cfg_get(cfg, "mpc.u_min", -1.0))
        self.u_max = float(cfg_get(cfg, "mpc.u_max", 1.0))
        self.horizon = float(cfg_get(cfg, "mpc.horizon_seconds", 0.8))
        self.iters = int(cfg_get(cfg, "mpc.iters", 120))
    def solve(self, phi0: float, omega: float, phi_target: float) -> float:
        N = max(2, int(self.horizon / self.dt))
        A = np.tril(np.ones((N, N)))
        c0 = phi0 + np.arange(1, N + 1) * (omega * self.dt)
        phi_ref = np.full(N, phi_target)
        H = (self.B ** 2) * (A.T @ A) + self.lambda_u * np.eye(N)
        D = np.zeros((N - 1, N))
        for i in range(N - 1):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0
        H += self.lambda_du * (D.T @ D)
        g = (self.B * A).T @ (c0 - phi_ref)
        u = np.zeros(N)
        L = float(np.linalg.eigvalsh(H).max() if N > 1 else 1.0)
        step = 1.0 / (L + 1e-6)
        for _ in range(self.iters):
            grad = H @ u + g
            u -= step * grad
            u = np.clip(u, self.u_min, self.u_max)
        return float(u[0])
class DopamineModel:
    #synthetic dopamine signal from rpe
    def __init__(self, cfg: Dict[str, Any]):
        self.baseline = float(cfg_get(cfg, "dopamine.baseline", 0.35))
        self.tau_s = float(cfg_get(cfg, "dopamine.tau_s", 2.0))
        self.k_reward = float(cfg_get(cfg, "dopamine.k_reward", 1.0))
        self.k_decay = float(cfg_get(cfg, "dopamine.k_decay", 0.5))
        self.rpe_smoothing = float(cfg_get(cfg, "dopamine.rpe_smoothing", 0.95))
        self.clip = float(cfg_get(cfg, "dopamine.clip", 2.0))
        self.value = self.baseline
        self.expected_reward = 0.0
    def update(self, reward: float, dt: float) -> Tuple[float, float]:
        self.expected_reward = (
            self.rpe_smoothing * self.expected_reward + (1.0 - self.rpe_smoothing) * reward
        )
        rpe = reward - self.expected_reward
        d = self.k_reward * rpe - self.k_decay * (self.value - self.baseline)
        self.value += (dt / max(1e-6, self.tau_s)) * d
        self.value = float(np.clip(self.value, 0.0, 1.5))
        rpe = float(np.clip(rpe, -self.clip, self.clip))
        return self.value, rpe
class RLPolicy:
    #adaptive policy using tanh whose graph is limite by +-1
    def __init__(self, cfg: Dict[str, Any]):
        self.enabled = bool(cfg_get(cfg, "rl.enabled", True))
        self.lr = float(cfg_get(cfg, "rl.learning_rate", 0.015))
        self.weight_decay = float(cfg_get(cfg, "rl.weight_decay", 1e-3))
        self.action_scale = float(cfg_get(cfg, "rl.action_scale", 0.8))
        self.exploration_sigma = float(cfg_get(cfg, "rl.exploration_sigma", 0.05))
        self.state_features = list(cfg_get(cfg, "rl.state_features", []))
        self.weights = np.zeros(len(self.state_features), dtype=np.float32)
        self.bias = 0.0
        self.prev_state = None
        self.prev_action = 0.0
        self.load_path = cfg_get(cfg, "rl.load_path", None)
        self.save_path = cfg_get(cfg, "rl.save_path", None)
        if self.load_path:
            self.load(self.load_path)
    def act(self, state_vec: np.ndarray) -> float:
        if not self.enabled:
            return 0.0
        z = float(np.dot(self.weights, state_vec) + self.bias)
        u = math.tanh(z) * self.action_scale
        if self.exploration_sigma > 0:
            u += float(np.random.normal(0.0, self.exploration_sigma))
        self.prev_state = state_vec
        self.prev_action = u
        return float(np.clip(u, -1.0, 1.0))
    def update(self, reward: float, rpe: float, dopamine: float) -> None:
        if not self.enabled or self.prev_state is None:
            return
        lr_eff = self.lr * (0.5 + dopamine)
        grad = self.prev_state
        self.weights += lr_eff * rpe * grad - self.weight_decay * self.weights
        self.bias += lr_eff * rpe
    def save(self, path: str) -> None:
        try:
            data = {"weights": self.weights.tolist(), "bias": self.bias}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            LOG.warning("failed to save RL state: %s", e)
    def load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.weights = np.array(data.get("weights", self.weights), dtype=np.float32)
            self.bias = float(data.get("bias", self.bias))
        except Exception as e:
            LOG.warning("failed to load RL state: %s", e)
#realtime pipeline
class RealtimePipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.stop_event = threading.Event()
        self.watchdog_s = float(cfg_get(cfg, "safety.watchdog_s", 0.2))
        self.last_packet_mono: Optional[float] = None
        self.watchdog_tripped = False
        self.ack_event = threading.Event()
        self.ack_time_ref: Dict[str, float] = {"mono": 0.0}
        self.reader: Optional[SerialFrameReader] = None
        self.sensor_bounds = cfg_get(cfg, "sensor_bounds", {})
        self.sample_q: "queue.Queue" = queue.Queue(maxsize=4096)
        self.emg_fs = float(cfg_get(cfg, "sampling.emg_fs", 860.0))
        self.imu_fs = float(cfg_get(cfg, "sampling.imu_fs", 200.0))
        self.window_s = float(cfg_get(cfg, "sampling.window_seconds", 0.5))
        self.control_dt = float(cfg_get(cfg, "control.interval_s", 0.05))
        self.decimator = Decimator(self.emg_fs / self.imu_fs)
        emg_len = max(32, int(self.emg_fs * self.window_s))
        imu_len = max(16, int(self.imu_fs * self.window_s))
        #rolling windows for speed
        self.emg_bp_buf = RingBuffer(emg_len)
        self.emg_env_buf = RingBuffer(emg_len)
        self.imu_buf = RingBuffer(imu_len)
        self.gx_buf = RingBuffer(imu_len)
        self.gy_buf = RingBuffer(imu_len)
        self.gz_buf = RingBuffer(imu_len)
        emg_band = cfg_get(cfg, "dsp.emg_band", [20.0, 250.0])
        imu_band = cfg_get(cfg, "dsp.imu_band", [3.0, 12.0])
        self.emg_filter = StreamingSOSFilter(self.emg_fs, (emg_band[0], emg_band[1]), kind="bandpass", order=4)
        self.imu_filter = StreamingSOSFilter(self.imu_fs, (imu_band[0], imu_band[1]), kind="bandpass", order=4)
        self.emg_env_filter = StreamingSOSFilter(
            self.emg_fs, float(cfg_get(cfg, "dsp.emg_env_lp_hz", 5.0)), kind="lowpass", order=2
        )
        self.axis_default = cfg_get(cfg, "dsp.axis_default", "gz")
        self.axis_autoselect = bool(cfg_get(cfg, "dsp.axis_autoselect", True))
        self.axis_selected = self.axis_default
        self.axis_history = {"gx": RingBuffer(imu_len), "gy": RingBuffer(imu_len), "gz": RingBuffer(imu_len)}
        model_path = cfg_get(cfg, "model.keras_h5", None)
        norm_path = cfg_get(cfg, "model.norm_np", None)
        self.infer = InferenceEngine(model_path, norm_path, int(cfg_get(cfg, "model.num_threads", 2)))
        self.safety = SafetyGuard(cfg)
        self.mpc = MPCControllerLite(cfg, self.control_dt)
        self.dopamine = DopamineModel(cfg)
        self.rl = RLPolicy(cfg)
        self.output = ControlOutput(cfg)
        if self.output.enabled:
            self.output.attach_ack(self.ack_event, self.ack_time_ref)
        logs_dir = cfg_get(cfg, "paths.logs_dir", "./logs")
        os.makedirs(logs_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(logs_dir, f"realtime_log_{ts}.csv")
        self.log_f = open(self.log_path, "w", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_f)
        self.log_writer.writerow([
            "time_s",
            "sensor_rx_ts",
            "inference_done_ts",
            "command_send_ts",
            "ack_rx_ts",
            "end_to_end_latency_ms",
            "emg_rms",
            "emg_env_mean",
            "emg_centroid_hz",
            "imu_amp_mean",
            "imu_phase_last",
            "imu_freq_hz",
            "imu_peak_hz",
            "emg_imu_corr",
            "plv_emg_imu",
            "tremor_power",
            "tremor_prob",
            "phase_pred",
            "amp_pred",
            "u_phase",
            "u_mpc",
            "u_rl",
            "u_final",
            "dopamine",
            "reward",
            "rpe",
            "latency_ms",
            "energy_j",
            "emergency_active",
        ])
        self.log_last = 0.0
        self.log_period = 1.0 / float(cfg_get(cfg, "logging.log_frequency_hz", 10))
        self.max_log_mb = float(cfg_get(cfg, "logging.max_log_file_size_mb", 50))
        self.max_log_bytes = int(self.max_log_mb * 1024 * 1024)
        self.backup_count = int(cfg_get(cfg, "logging.backup_count", 5))
        hw = cfg_get(cfg, "hardware", {})
        self.adc_bits = int(hw.get("emg_adc_bits", 12))
        self.adc_offset = int(hw.get("emg_adc_offset", 2048))
        self.vref = float(hw.get("emg_vref", 3.3))
        self.emg_lsb = self.vref / (2 ** self.adc_bits)
        self.gyro_scale = float(hw.get("gyro_scale_per_lsb", 0.01))
        self.amp_baseline = EWMA(
            cfg_get(cfg, "metrics.amp_baseline_tau_s", 5.0),
            self.control_dt,
            initial=0.0,
        )
    def close(self) -> None:
        try:
            self.log_f.flush()
            self.log_f.close()
        except Exception:
            pass
        if self.rl.save_path:
            self.rl.save(self.rl.save_path)
        self.output.close()
        if self.reader is not None:
            self.reader.close()
    def fail_safe_shutdown(self) -> None:
        try:
            self.output.safe_disable()
        except Exception:
            pass
        if self.reader is not None:
            self.reader.send_raw("STIM,0,0,0\n")
        self.stop_event.set()
    def rotate_logs_if_needed(self) -> None:
        if self.log_f.tell() < self.max_log_bytes:
            return
        self.log_f.close()
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.log_path}.{i}"
            dst = f"{self.log_path}.{i + 1}"
            if os.path.exists(src):
                os.rename(src, dst)
        os.rename(self.log_path, f"{self.log_path}.1")
        self.log_f = open(self.log_path, "w", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_f)
        self.log_writer.writerow([
            "time_s",
            "sensor_rx_ts",
            "inference_done_ts",
            "command_send_ts",
            "ack_rx_ts",
            "end_to_end_latency_ms",
            "emg_rms",
            "emg_env_mean",
            "emg_centroid_hz",
            "imu_amp_mean",
            "imu_phase_last",
            "imu_freq_hz",
            "imu_peak_hz",
            "emg_imu_corr",
            "plv_emg_imu",
            "tremor_power",
            "tremor_prob",
            "phase_pred",
            "amp_pred",
            "u_phase",
            "u_mpc",
            "u_rl",
            "u_final",
            "dopamine",
            "reward",
            "rpe",
            "latency_ms",
            "energy_j",
            "emergency_active",
        ])
    def _select_axis_if_ready(self) -> None:
        if not self.axis_autoselect:
            return
        if len(self.axis_history["gx"]) < self.imu_buf.size:
            return
        imu_band = cfg_get(self.cfg, "dsp.imu_band", [3.0, 12.0])
        powers = {
            k: band_power(buf.get(), self.imu_fs, (imu_band[0], imu_band[1]))
            for k, buf in self.axis_history.items()
        }
        self.axis_selected = max(powers, key=powers.get)
        self.axis_autoselect = False
        LOG.info("Auto-selected IMU axis: %s (powers=%s)", self.axis_selected, powers)
    def _build_model_input(self, features: Dict[str, float], signals: Dict[str, np.ndarray]) -> Any:
        if not self.infer.input_shapes:
            return np.zeros((1, 1), dtype=np.float32)
        input_shapes = self.infer.input_shapes
        input_shape = input_shapes[0]
        input_type = cfg_get(self.cfg, "model.input_type", "auto")
        #from cnn lstm input emg sequence and imu sequence
        if len(input_shapes) == 2:
            def _resample_1d(arr: np.ndarray, target_t: int) -> np.ndarray:
                if len(arr) == target_t:
                    return arr.astype(np.float32)
                if len(arr) <= 1:
                    return np.zeros(target_t, dtype=np.float32)
                return np.interp(
                    np.linspace(0, 1, target_t),
                    np.linspace(0, 1, len(arr)),
                    arr,
                ).astype(np.float32)
            emg_src = np.asarray(signals.get("emg_bp", np.zeros(1)), dtype=np.float32)
            imu_src = np.asarray(signals.get("imu_bp", np.zeros(1)), dtype=np.float32)
            emg_t = int(input_shapes[0][1]) if len(input_shapes[0]) >= 3 and input_shapes[0][1] not in (None, 0) else len(emg_src)
            imu_t = int(input_shapes[1][1]) if len(input_shapes[1]) >= 3 and input_shapes[1][1] not in (None, 0) else len(imu_src)
            emg_in = _resample_1d(emg_src, max(1, emg_t)).reshape(1, max(1, emg_t), 1)
            imu_in = _resample_1d(imu_src, max(1, imu_t)).reshape(1, max(1, imu_t), 1)
            return [emg_in.astype(np.float32), imu_in.astype(np.float32)]
        #feature model just build a vector of features in the specified order
        if input_type == "features" or (input_type == "auto" and len(input_shape) == 2):
            order = cfg_get(self.cfg, "model.feature_order", [])
            vec = np.array([features.get(k, 0.0) for k in order], dtype=np.float32)
            return vec.reshape(1, -1)
        #Sequence model
        seq_signals = cfg_get(self.cfg, "model.sequence_signals", [])
        base = signals.get(seq_signals[0], np.zeros(1))
        T = len(base)
        channels = []
        for name in seq_signals:
            arr = signals.get(name, np.zeros(T))
            if len(arr) != T:
                arr = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(arr)), arr)
            channels.append(arr.astype(np.float32))
        X = np.stack(channels, axis=-1)
        target_T = None
        if len(input_shape) == 3:
            if input_shape[1] not in (None, 0) and input_shape[1] != X.shape[0]:
                target_T = int(input_shape[1])
            elif input_shape[2] not in (None, 0) and input_shape[2] != X.shape[0]:
                target_T = int(input_shape[2])
        if target_T and target_T > 1:
            X = np.stack(
                [np.interp(np.linspace(0, 1, target_T), np.linspace(0, 1, X.shape[0]), X[:, c]) for c in range(X.shape[1])],
                axis=-1,
            )
        ds = max(1, int(cfg_get(self.cfg, "model.sequence_downsample", 1)))
        if ds > 1:
            X = X[::ds]
        if len(input_shape) == 3 and input_shape[1] == X.shape[1]:
            X = np.transpose(X, (1, 0))
        return X.astype(np.float32)[None, ...]
    def _parse_model_outputs(self, pred: Any) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if pred is None:
            return out
        vec = np.zeros(0, dtype=float)
        if isinstance(pred, list):
            for name, arr in zip(self.infer.output_names, pred):
                vals = np.ravel(np.asarray(arr)).astype(float)
                lname = name.lower()
                if "amp" in lname and len(vals) >= 1:
                    out["amp_pred"] = float(vals[0])
                elif "phase" in lname and len(vals) >= 2:
                    out["phase_pred"] = float(math.atan2(vals[1], vals[0]))
                elif "tremor" in lname and len(vals) >= 1:
                    out["tremor_prob"] = float(vals[0])
            if not out and pred:
                vec = np.concatenate([np.ravel(np.asarray(a)).astype(float) for a in pred])
        else:
            vec = np.ravel(np.asarray(pred)).astype(float)
        names = cfg_get(self.cfg, "model.output_map", [])
        for i, name in enumerate(names):
            if name in out:
                continue
            if i < len(vec):
                val = float(vec[i])
                activation = cfg_get(self.cfg, f"model.output_activation.{name}", None)
                if activation == "sigmoid":
                    val = 1.0 / (1.0 + math.exp(-val))
                out[name] = val
        return out
    def _compute_features(self, emg_bp: np.ndarray, emg_env: np.ndarray, imu_bp: np.ndarray) -> Dict[str, float]:
        emg_rms = float(np.sqrt(np.mean(emg_bp ** 2) + 1e-12))
        emg_env_mean = float(np.mean(emg_env))
        emg_centroid, _ = spectral_features(emg_bp, self.emg_fs)
        analytic = analytic_signal(imu_bp)
        imu_amp = np.abs(analytic)
        imu_phase = np.angle(analytic)
        imu_amp_mean = float(np.mean(imu_amp))
        imu_phase_last = float(imu_phase[-1])
        imu_phase_slope = phase_slope(imu_phase, self.imu_fs)
        imu_freq_hz = float(imu_phase_slope / (2.0 * math.pi))
        _, imu_peak_hz = spectral_features(imu_bp, self.imu_fs)
        emg_imu_corr = corrcoef_safe(emg_env, imu_amp)
        plv_val = plv(np.angle(analytic_signal(emg_env)), imu_phase)
        imu_band = cfg_get(self.cfg, "dsp.imu_band", [3.0, 12.0])
        tremor_power = band_power(imu_bp, self.imu_fs, (imu_band[0], imu_band[1]))
        return {
            "emg_rms": emg_rms,
            "emg_env_mean": emg_env_mean,
            "emg_centroid_hz": emg_centroid,
            "imu_amp_mean": imu_amp_mean,
            "imu_phase_last": imu_phase_last,
            "imu_freq_hz": imu_freq_hz,
            "imu_peak_hz": imu_peak_hz,
            "emg_imu_corr": emg_imu_corr,
            "plv_emg_imu": plv_val,
            "tremor_power": tremor_power,
        }
    def _estimate_tremor_prob(self, features: Dict[str, float], pred_out: Dict[str, float]) -> float:
        if "tremor_prob" in pred_out:
            return float(pred_out["tremor_prob"])
        amp = features["imu_amp_mean"]
        freq = features["imu_freq_hz"]
        amp_thresh = cfg_get(self.cfg, "prediction.amp_thresh", 0.15)
        freq_lo, freq_hi = cfg_get(self.cfg, "prediction.freq_range_hz", [3.0, 8.0])
        amp_score = 1.0 / (1.0 + math.exp(-(amp - amp_thresh) * 8.0))
        freq_score = 1.0 if freq_lo <= freq <= freq_hi else 0.0
        return float(np.clip(amp_score * (0.5 + 0.5 * freq_score), 0.0, 1.0))
    def _predict_phase_amp(self, features: Dict[str, float], pred_out: Dict[str, float]) -> Tuple[float, float]:
        phase = features["imu_phase_last"]
        amp = features["imu_amp_mean"]
        if "phase_pred" in pred_out:
            phase_pred = float(pred_out["phase_pred"])
        else:
            horizon = cfg_get(self.cfg, "prediction.phase_horizon_s", 0.2)
            omega = features["imu_freq_hz"] * 2.0 * math.pi
            phase_pred = phase + omega * horizon
        if "amp_pred" in pred_out:
            amp_pred = float(pred_out["amp_pred"])
        else:
            amp_pred = amp
        return phase_pred, amp_pred
    def _build_rl_state(self, features: Dict[str, float], tremor_prob: float) -> np.ndarray:
        phase = features["imu_phase_last"]
        phase_target = phase + cfg_get(self.cfg, "control.phase_offset_rad", math.pi)
        phase_error = math.sin(phase - phase_target)
        state_map = {
            "imu_amp_mean": features["imu_amp_mean"],
            "imu_freq_hz": features["imu_freq_hz"],
            "emg_rms": features["emg_rms"],
            "phase_error": phase_error,
            "tremor_prob": tremor_prob,
            "dopamine": self.dopamine.value,
        }
        return np.array([state_map.get(k, 0.0) for k in self.rl.state_features], dtype=np.float32)
    def _compute_control(self, features: Dict[str, float], tremor_prob: float, phase_pred: float) -> Tuple[float, float, float]:
        mode = cfg_get(self.cfg, "control.mode", "hybrid")
        phase = features["imu_phase_last"]
        amp = features["imu_amp_mean"]
        omega = features["imu_freq_hz"] * 2.0 * math.pi
        phase_target = phase + cfg_get(self.cfg, "control.phase_offset_rad", math.pi)
        k_phase = cfg_get(self.cfg, "control.k_phase", 0.8)
        k_amp = cfg_get(self.cfg, "control.k_amp", 0.4)
        u_phase = k_phase * math.sin(phase - phase_target) + k_amp * amp
        u_mpc = 0.0
        if mode in ("mpc", "hybrid"):
            u_mpc = self.mpc.solve(phase_pred, omega, phase_target)
        u_base = u_phase if mode == "phase_lock" else u_mpc
        if mode == "hybrid":
            mpc_blend = cfg_get(self.cfg, "control.mpc_blend", 0.65)
            u_base = (1.0 - mpc_blend) * u_phase + mpc_blend * u_mpc
        u_rl = 0.0
        if cfg_get(self.cfg, "rl.enabled", True):
            state = self._build_rl_state(features, tremor_prob)
            u_rl = self.rl.act(state)
            rl_blend = cfg_get(self.cfg, "control.rl_blend", 0.35)
            blend = rl_blend * (0.5 + 0.5 * tremor_prob)
            u_final = (1.0 - blend) * u_base + blend * u_rl
        else:
            u_final = u_base
        u_min = cfg_get(self.cfg, "control.u_min", -1.0)
        u_max = cfg_get(self.cfg, "control.u_max", 1.0)
        return (
            float(np.clip(u_phase, u_min, u_max)),
            float(np.clip(u_mpc, u_min, u_max)),
            float(np.clip(u_final, u_min, u_max)),
        )
    def _compute_reward(self, amp: float, latency_ms: float, energy_j: float) -> float:
        baseline = max(1e-6, self.amp_baseline.value)
        amp_reduction = (baseline - amp) / baseline
        amp_reduction = float(np.clip(amp_reduction, -1.0, 1.0))
        weights = cfg_get(self.cfg, "metrics.reward_weights", {"amp": 1.0, "energy": 0.4, "latency": 0.2})
        target_lat = cfg_get(self.cfg, "metrics.latency_target_ms", 50.0)
        latency_penalty = max(0.0, (latency_ms - target_lat) / target_lat)
        reward = (
            weights.get("amp", 1.0) * amp_reduction - weights.get("energy", 0.4) * energy_j - weights.get("latency", 0.2) * latency_penalty
        )
        return float(reward)
    def _valid_sample(self, emg: float, gx: float, gy: float, gz: float) -> bool:
        if not (np.isfinite(emg) and np.isfinite(gx) and np.isfinite(gy) and np.isfinite(gz)):
            return False
        bounds = self.sensor_bounds
        def in_bounds(v: float, key: str) -> bool:
            lo, hi = bounds.get(key, (-1e12, 1e12))
            return lo <= v <= hi
        return in_bounds(emg, "emg") and in_bounds(gx, "gx") and in_bounds(gy, "gy") and in_bounds(gz, "gz")
    def run(self) -> None:
        debug = cfg_get(self.cfg, "debug.enable_debug_mode", False)
        if debug:
            LOG.setLevel(getattr(logging, cfg_get(self.cfg, "debug.debug_level", "INFO"), logging.INFO))
        if cfg_get(self.cfg, "debug.simulate_serial_data", False):
            sim_path = cfg_get(self.cfg, "debug.simulated_data_file", "simulated_data.csv")
            sim_data = load_simulated_csv(sim_path)
            reader: threading.Thread = SimulatedReader(
                sim_data,
                self.sample_q,
                self.stop_event,
                bool(cfg_get(self.cfg, "debug.simulate_real_time", True)),
                float(cfg_get(self.cfg, "debug.simulation_speed_factor", 1.0)),
            )
        else:
            self.reader = SerialFrameReader(
                cfg_get(self.cfg, "serial.port", "/dev/ttyACM0"),
                int(cfg_get(self.cfg, "serial.baud", 2000000)),
                self.sample_q,
                self.stop_event,
                self.emg_fs,
                ack_event=self.ack_event,
                ack_time_ref=self.ack_time_ref,
            )
            if self.output.enabled and self.output.shared_serial:
                self.output.attach_sender(self.reader.send_raw)
            reader = self.reader
        reader.start()
        LOG.info("realtime pipeline started")
        next_control_mono = time.monotonic()
        try:
            while not self.stop_event.is_set():
                t_s = None
                try:
                    t_us, emg_raw, gx, gy, gz = self.sample_q.get(timeout=0.05)
                except queue.Empty:
                    t_us = emg_raw = gx = gy = gz = None
                now_mono = time.monotonic()
                if t_us is not None:
                    if self._valid_sample(float(emg_raw), float(gx), float(gy), float(gz)):
                        self.last_packet_mono = now_mono
                        t_s = t_us * 1e-6
                        emg_v = (float(emg_raw) - self.adc_offset) * self.emg_lsb
                        emg_bp = self.emg_filter.step(emg_v)
                        emg_env = self.emg_env_filter.step(abs(emg_bp))
                        self.emg_bp_buf.append(emg_bp)
                        self.emg_env_buf.append(emg_env)
                        if self.decimator.step():
                            gx_s = float(gx) * self.gyro_scale
                            gy_s = float(gy) * self.gyro_scale
                            gz_s = float(gz) * self.gyro_scale
                            self.axis_history["gx"].append(gx_s)
                            self.axis_history["gy"].append(gy_s)
                            self.axis_history["gz"].append(gz_s)
                            self.gx_buf.append(gx_s)
                            self.gy_buf.append(gy_s)
                            self.gz_buf.append(gz_s)
                            self._select_axis_if_ready()
                            imu_axis = {"gx": gx_s, "gy": gy_s, "gz": gz_s}[self.axis_selected]
                            imu_bp = self.imu_filter.step(imu_axis)
                            self.imu_buf.append(imu_bp)
                #watchdog for stopping stim if packets stop coming in
                if self.last_packet_mono is None or (now_mono - self.last_packet_mono) > self.watchdog_s:
                    if not self.watchdog_tripped:
                        self.output.safe_disable()
                        if not self.output.enabled and self.reader is not None:
                            self.reader.send_raw("STIM,0,0,0\n")
                        self.watchdog_tripped = True
                    continue
                if self.watchdog_tripped:
                    self.watchdog_tripped = False
                #fixed-rate control tick
                if now_mono < next_control_mono:
                    continue
                while next_control_mono <= now_mono:
                    next_control_mono += self.control_dt
                control_start = time.time()
                emg_bp_win = self.emg_bp_buf.get()
                emg_env_win = self.emg_env_buf.get()
                imu_bp_win = self.imu_buf.get()
                if len(emg_bp_win) < 8 or len(imu_bp_win) < 8:
                    continue
                features = self._compute_features(emg_bp_win, emg_env_win, imu_bp_win)
                gx_win = self.gx_buf.get()
                gy_win = self.gy_buf.get()
                gz_win = self.gz_buf.get()
                if len(gx_win) < 8 or len(gy_win) < 8 or len(gz_win) < 8:
                    continue
                ax_win = np.zeros_like(gx_win)
                ay_win = np.zeros_like(gy_win)
                az_win = np.zeros_like(gz_win)
                signals = {
                    "emg_bp": emg_bp_win,
                    "emg_env": emg_env_win,
                    "imu_bp": imu_bp_win,
                    "imu_amp": np.abs(analytic_signal(imu_bp_win)),
                    "imu_phase": np.angle(analytic_signal(imu_bp_win)),
                    "ax": ax_win,
                    "ay": ay_win,
                    "az": az_win,
                    "gx": gx_win,
                    "gy": gy_win,
                    "gz": gz_win,
                }
                model_input = self._build_model_input(features, signals)
                model_input = self.infer.apply_norm(model_input)
                pred = self.infer.predict(model_input)
                inference_done_ts = time.monotonic()
                pred_out = self._parse_model_outputs(pred)
                tremor_prob = self._estimate_tremor_prob(features, pred_out)
                phase_pred, amp_pred = self._predict_phase_amp(features, pred_out)
                u_phase, u_mpc, u_final = self._compute_control(features, tremor_prob, phase_pred)
                latency_ms = (time.time() - control_start) * 1000.0
                energy_scale = cfg_get(self.cfg, "metrics.energy_scale", 0.06)
                energy_j = energy_scale * (u_final ** 2) * self.control_dt
                self.amp_baseline.update(features["imu_amp_mean"])
                reward = self._compute_reward(features["imu_amp_mean"], latency_ms, energy_j)
                dopamine, rpe = self.dopamine.update(reward, self.control_dt)
                self.rl.update(reward, rpe, dopamine)
                u_safe, emergency = self.safety.apply(u_final, features["imu_amp_mean"], now_mono)
                cmd_send_ts = time.monotonic()
                ack_ok = self.output.send(u_safe, features["imu_phase_last"], features["imu_amp_mean"])
                ack_ts = self.ack_time_ref.get("mono", 0.0) if ack_ok else 0.0
                end_to_end_ms = 0.0
                if self.last_packet_mono is not None:
                    if ack_ts and ack_ts > 0:
                        end_to_end_ms = (ack_ts - self.last_packet_mono) * 1000.0
                    else:
                        end_to_end_ms = (cmd_send_ts - self.last_packet_mono) * 1000.0
                if time.time() - self.log_last >= self.log_period:
                    self.log_last = time.time()
                    self.log_writer.writerow([
                        f"{t_s:.6f}" if t_s is not None else "",
                        f"{self.last_packet_mono:.6f}" if self.last_packet_mono else "",
                        f"{inference_done_ts:.6f}",
                        f"{cmd_send_ts:.6f}",
                        f"{ack_ts:.6f}" if ack_ts else "",
                        f"{end_to_end_ms:.2f}",
                        f"{features['emg_rms']:.6f}",
                        f"{features['emg_env_mean']:.6f}",
                        f"{features['emg_centroid_hz']:.4f}",
                        f"{features['imu_amp_mean']:.6f}",
                        f"{features['imu_phase_last']:.6f}",
                        f"{features['imu_freq_hz']:.4f}",
                        f"{features['imu_peak_hz']:.4f}",
                        f"{features['emg_imu_corr']:.4f}",
                        f"{features['plv_emg_imu']:.4f}",
                        f"{features['tremor_power']:.6f}",
                        f"{tremor_prob:.4f}",
                        f"{phase_pred:.6f}",
                        f"{amp_pred:.6f}",
                        f"{u_phase:.6f}",
                        f"{u_mpc:.6f}",
                        f"{self.rl.prev_action if self.rl.enabled else 0.0:.6f}",
                        f"{u_safe:.6f}",
                        f"{dopamine:.4f}",
                        f"{reward:.4f}",
                        f"{rpe:.4f}",
                        f"{latency_ms:.2f}",
                        f"{energy_j:.6f}",
                        int(emergency),
                    ])
                    self.rotate_logs_if_needed()
        finally:
            self.close()
def setup_logging(debug_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, debug_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime neuroadaptive EMG+IMU pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML/JSON config")
    parser.add_argument("--port", type=str, default=None, help="Serial port override")
    parser.add_argument("--baud", type=int, default=None, help="Serial baud override")
    parser.add_argument("--model", type=str, default=None, help="Model path override")
    parser.add_argument("--norm", type=str, default=None, help="Norm path override")
    parser.add_argument("--models-dir", type=str, default=TRAINING_MODELS_DIR, help="Training models folder")
    parser.add_argument("--logdir", type=str, default=None, help="Logs dir override")
    parser.add_argument("--simulate", action="store_true", help="Use simulated CSV input")
    args = parser.parse_args()
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    cfg.setdefault("paths", {})["models_dir"] = args.models_dir
    if args.port:
        cfg.setdefault("serial", {})["port"] = args.port
    if args.baud is not None:
        cfg.setdefault("serial", {})["baud"] = args.baud
    if args.model:
        cfg.setdefault("model", {})["keras_h5"] = args.model
    if args.norm:
        cfg.setdefault("model", {})["norm_np"] = args.norm
    if args.logdir:
        cfg.setdefault("paths", {})["logs_dir"] = args.logdir
    if args.simulate:
        cfg.setdefault("debug", {})["simulate_serial_data"] = True
    #get the tuned mpc and training artifacts
    cfg = apply_training_models_defaults(cfg)
    setup_logging(cfg_get(cfg, "debug.debug_level", "INFO"))
    pipeline = RealtimePipeline(cfg)
    def _sig_handler(signum, frame) -> None:
        LOG.info("Stopping (signal %s)", signum)
        pipeline.fail_safe_shutdown()
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.fail_safe_shutdown()
    except Exception as e:
        LOG.exception("Fatal error: %s", e)
        pipeline.fail_safe_shutdown()

if __name__ == "__main__":
    main()
