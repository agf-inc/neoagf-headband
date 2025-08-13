#!/usr/bin/env python3

import asyncio
import logging
import struct
from datetime import datetime
from bleak import BleakClient, BleakScanner
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Configuration
DEVICE_NAME = "ESP32_EEG"  # Match the name your ESP32 is advertising
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
SCAN_TIMEOUT = 10.0
SAMPLE_RATE_HZ = 250  # EEG sample rate
WINDOW_SECONDS = 5     # Sliding window length for analysis
REPORT_INTERVAL = 1.0/2  # Seconds between reports
HIGH_FOCUS_THRESHOLD = 0.55
LOW_FOCUS_THRESHOLD = 0.35  

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Script Configuration ---
DEBUG_MODE = False # Set to True to see continuous data logging

# Configuration for slope-based event detection
# These values represent the change in relative power per second.
# You will likely need to tune these based on your data.
SLOPE_THRESHOLDS = {
    "delta_increase": 0.1,    # Threshold for a sudden increase in delta power
    "delta_decrease": -0.1,   # Threshold for a sudden decrease in delta power
    "theta_increase": 0.1,    # Threshold for a sudden increase in theta power
    "theta_decrease": -0.1,   # Threshold for a sudden decrease in theta power
    "focus_increase": 0.1,    # Threshold for a sudden increase in beta or gamma
    "history_len": 4,         # How many data points to use for slope calculation
    "cooldown_s": 2.0,        # Cooldown period after detecting an event
}

class EEGDataCollector:
    def __init__(self):
        self.data_count = 0
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.filename = f"eeg_data_{timestamp}.csv"
        with open(self.filename, "w") as f:
            f.write("timestamp,sample_number,eeg_value\n")
        logger.info(f"Data will be saved to: {self.filename}")
        # Buffer for recent samples for analysis
        self.buffer = deque(maxlen=WINDOW_SECONDS * SAMPLE_RATE_HZ)
        self._running = True
        
        # State inference attributes
        self.current_state = "Stable"
        self.state_change_cooldown = 0
        hist_len = SLOPE_THRESHOLDS["history_len"]
        self.time_history = deque(maxlen=hist_len)
        self.delta_history = deque(maxlen=hist_len)
        self.theta_history = deque(maxlen=hist_len)
        self.beta_history = deque(maxlen=hist_len)
        self.gamma_history = deque(maxlen=hist_len)
        self.event_counts = {"Eyes Closed Event": 0, "Eyes Open Event": 0, "Low Focus Event": 0, "High Focus Event": 0}

        # Define EEG bands (Hz), all < 100 Hz
        self.bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 90.0),  # keep under 100 Hz
        }

        # Histories for dashboard
        self.hist_maxlen = 60 * int(1 / REPORT_INTERVAL)  # keep last ~60 seconds
        self.t_hist = deque(maxlen=self.hist_maxlen)
        self.band_hist = {k: deque(maxlen=self.hist_maxlen) for k in self.bands}
        self.stats_hist = {k: deque(maxlen=self.hist_maxlen) for k in ("vmin", "vmax", "vavg", "vrms")}
        
        # Event timeline history
        self.event_times = []  # List of (time, event_name) tuples
        self.event_maxlen = 100  # Keep last 100 events

        # Matplotlib objects
        self._fig = None
        self._axes = None
        self._lines_bands = {}
        self._lines_stats = {}
        self._state_text = None

    def handle_notification(self, sender, data):
        try:
            # Handle both float or string transmission
            if len(data) == 4:
                eeg_value = struct.unpack('f', data)[0]
            else:
                eeg_value = float(data.decode().strip())

            self.data_count += 1
            now = datetime.now()
            elapsed = (now - self.start_time).total_seconds()

            # Save to CSV
            with open(self.filename, "a") as f:
                f.write(f"{now.isoformat()},{self.data_count},{eeg_value:.6f}\n")

            # Add to analysis buffer (do not print every sample)
            self.buffer.append(eeg_value)

        except Exception as e:
            logger.error(f"Notification error: {e}")

    def _compute_band_powers(self, data: np.ndarray):
        """Compute relative power in defined EEG bands using FFT-based PSD."""
        if data.size < 2:
            return {k: float('nan') for k in self.bands}

        # Detrend and windowing
        x = data - np.mean(data)
        window = np.hanning(x.size)
        xw = x * window

        # Notch filter powerline interference at 50 Hz and 60 Hz
        # Apply both (harmless if absent), zero-phase for minimal distortion
        b50, a50 = signal.iirnotch(50.0, 30.0, SAMPLE_RATE_HZ)
        b60, a60 = signal.iirnotch(60.0, 30.0, SAMPLE_RATE_HZ)
        xw = signal.filtfilt(b50, a50, xw)
        xw = signal.filtfilt(b60, a60, xw)

        # FFT
        freqs = np.fft.rfftfreq(xw.size, d=1.0 / SAMPLE_RATE_HZ)
        fft_vals = np.fft.rfft(xw)
        # Power spectral density (proportional)
        psd = (np.abs(fft_vals) ** 2)
        # Avoid DC dominance by ignoring 0 Hz in band calc except if band includes it
        total_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 100.0)])
        if total_power <= 0:
            total_power = np.finfo(float).eps

        band_powers = {}
        for name, (f_lo, f_hi) in self.bands.items():
            mask = (freqs >= f_lo) & (freqs < f_hi)
            band_power = np.sum(psd[mask])
            band_powers[name] = float(band_power / total_power)
        return band_powers

    def _compute_signal_stats(self, data: np.ndarray):
        if data.size == 0:
            return {"vmin": float('nan'), "vmax": float('nan'), "vavg": float('nan'), "vrms": float('nan')}
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        vavg = float(np.mean(data))
        vrms = float(np.sqrt(np.mean(np.square(data))))
        return {"vmin": vmin, "vmax": vmax, "vavg": vavg, "vrms": vrms}

    async def periodic_report(self):
        """Periodically print raw signal stats and band powers from recent buffer."""
        last_count = 0
        while self._running:
            await asyncio.sleep(REPORT_INTERVAL)
            # Copy buffer to numpy array
            data = np.array(self.buffer, dtype=float)
            if data.size < SLOPE_THRESHOLDS["history_len"]:
                continue

            n = data.size
            duration = n / SAMPLE_RATE_HZ
            stats = self._compute_signal_stats(data)
            bands = self._compute_band_powers(data)
            
            # --- State Inference based on slope ---
            t_now = (datetime.now() - self.start_time).total_seconds()
            self.time_history.append(t_now)
            self.delta_history.append(bands.get("delta", 0))
            self.theta_history.append(bands.get("theta", 0))
            self.beta_history.append(bands.get("beta", 0))
            self.gamma_history.append(bands.get("gamma", 0))

            new_state = "Stable"
            if self.state_change_cooldown > 0:
                self.state_change_cooldown -= REPORT_INTERVAL
            
            # Check for events if history is full and not in cooldown
            if len(self.time_history) == self.time_history.maxlen and self.state_change_cooldown <= 0:
                # Calculate slopes
                delta_slope = np.polyfit(self.time_history, self.delta_history, 1)[0]
                theta_slope = np.polyfit(self.time_history, self.theta_history, 1)[0]
                beta_slope = np.polyfit(self.time_history, self.beta_history, 1)[0]
                gamma_slope = np.polyfit(self.time_history, self.gamma_history, 1)[0]

                # Apply rules
                # First check if Delta or Theta is dominant (relaxed/drowsy state)
                delta_power = bands.get("delta", 0)
                theta_power = bands.get("theta", 0)
                alpha_power = bands.get("alpha", 0)
                beta_power = bands.get("beta", 0)
                gamma_power = bands.get("gamma", 0)
                
                is_delta_dominant = delta_power > 0.3 and delta_power > alpha_power and delta_power > beta_power and delta_power > gamma_power and delta_power > theta_power
                is_gamma_dominant = gamma_power > 0.3 and gamma_power > alpha_power and gamma_power > beta_power and gamma_power > delta_power and gamma_power > theta_power
                is_theta_dominant = theta_power > 0.3 and theta_power > alpha_power and theta_power > beta_power and theta_power > gamma_power and theta_power > delta_power
                is_beta_dominant = beta_power > 0.3 and beta_power > alpha_power and beta_power > gamma_power and beta_power > delta_power and beta_power > theta_power
                is_delta_or_theta_dominant = is_delta_dominant or is_theta_dominant
                
                # Only check for eye states if Delta or Theta is dominant
                if is_delta_or_theta_dominant:
                #if is_delta_dominant:
                    # Check for Eyes Closed: Either Delta OR Theta slope is increasing above threshold
                    delta_increasing = delta_slope > SLOPE_THRESHOLDS["delta_increase"]
                    theta_increasing = theta_slope > SLOPE_THRESHOLDS["theta_increase"]
                    
                    if delta_increasing or theta_increasing:
                    #if delta_increasing :
                        new_state = "Eyes Closed Event"
                    # Check for Eyes Open: Either Delta OR Theta slope is decreasing below threshold
                    ## OLD Eyes Open detection (commented out for fallback)
                    #elif delta_slope < SLOPE_THRESHOLDS["delta_decrease"] or theta_slope < SLOPE_THRESHOLDS["theta_decrease"]:
                    #    new_state = "Eyes Open Event"
                    
                elif delta_slope < SLOPE_THRESHOLDS["delta_decrease"] or theta_slope < SLOPE_THRESHOLDS["theta_decrease"]:
                    new_state = "Eyes Open Event"
                    # NEW Eyes Open detection: Delta decreasing AND Gamma increasing (intersection pattern)
                #elif delta_slope < -0.02 and gamma_slope > 0.02 and gamma_power > 1.1 * delta_power :
                    ## OLD Eyes Open detection (commented out for fallback)
                    # elif delta_slope < SLOPE_THRESHOLDS["delta_decrease"] or theta_slope < SLOPE_THRESHOLDS["theta_decrease"]:
                #    new_state = "Eyes Open Event"
                # Check for Focus: Beta or Gamma slope is increasing
                #elif beta_slope > SLOPE_THRESHOLDS["focus_increase"] or gamma_slope > SLOPE_THRESHOLDS["focus_increase"]:
                #    new_state = "High Focus Event"
                #elif beta_slope > SLOPE_THRESHOLDS["medium_focus"] or gamma_slope > SLOPE_THRESHOLDS["medium_focus"]:
                #    new_state = "Medium Focus Event" 
                # Check for Focus: Beta or Gamma slope is increasing
                #elif is_gamma_dominant or is_beta_dominant:
                elif is_gamma_dominant:
                    #if beta_slope > SLOPE_THRESHOLDS["focus_increase"] or gamma_slope > SLOPE_THRESHOLDS["focus_increase"]:
                    #if gamma_slope > SLOPE_THRESHOLDS["focus_increase"]:
                    if gamma_power > HIGH_FOCUS_THRESHOLD :
                        new_state = "High Focus Event"
                    elif gamma_power > LOW_FOCUS_THRESHOLD :
                        new_state = "Low Focus Event"
                
                if new_state != "Stable":
                    self.state_change_cooldown = SLOPE_THRESHOLDS["cooldown_s"]
                    self.event_counts[new_state] += 1
                    # Add event to timeline
                    self.event_times.append((t_now, new_state))
                    # Keep only recent events
                    if len(self.event_times) > self.event_maxlen:
                        self.event_times.pop(0)

            self.current_state = new_state
            
            new_samples = self.data_count - last_count
            last_count = self.data_count

            # Log output based on DEBUG_MODE
            if DEBUG_MODE:
                logger.info(
                    (
                        "Stats over last %.1fs (%d samples, +%d): "
                        "Vmin=%.6f V, Vmax=%.6f V, Vavg=%.6f V, Vrms=%.6f V | "
                        "Bands (rel power): delta=%.2f, theta=%.2f, alpha=%.2f, beta=%.2f, gamma=%.2f | State: %s"
                    ),
                    duration, n, new_samples,
                    stats["vmin"], stats["vmax"], stats["vavg"], stats["vrms"],
                    bands.get("delta", 0), bands.get("theta", 0), bands.get("alpha", 0),
                    bands.get("beta", 0), bands.get("gamma", 0), self.current_state
                )
            elif self.current_state != "Stable":
                logger.info(f"EVENT DETECTED: {self.current_state}")

            # Append to histories (time in seconds since start)
            self.t_hist.append(t_now)
            for name in self.bands:
                self.band_hist[name].append(bands.get(name, float('nan')))
            for key in self.stats_hist:
                self.stats_hist[key].append(stats.get(key, float('nan')))

    def _init_dashboard(self):
        if self._fig is not None:
            return
        plt.ion()
        self._fig, self._axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        ax_bands, ax_stats, ax_events = self._axes

        # Band powers subplot
        colors = {
            "delta": "tab:blue",
            "theta": "tab:orange",
            "alpha": "tab:green",
            "beta": "tab:red",
            "gamma": "tab:purple",
        }
        for name in self.bands:
            (line,) = ax_bands.plot([], [], label=name, color=colors.get(name, None))
            self._lines_bands[name] = line
        ax_bands.set_ylabel("Rel Power")
        ax_bands.set_ylim(0, 1)
        ax_bands.set_title("EEG Band Powers (relative)")
        ax_bands.legend(loc="upper right")

        # State text
        self._state_text = ax_bands.text(
            0.5, 1.05, "", transform=ax_bands.transAxes, ha="center", va="center",
            fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray')
        )

        # Stats subplot
        for key, color in zip(["vmin", "vmax", "vavg", "vrms"], ["C0", "C3", "C2", "C1"]):
            (line,) = ax_stats.plot([], [], label=key, color=color)
            self._lines_stats[key] = line
        ax_stats.set_ylabel("Voltage (V)")
        ax_stats.set_title("Signal Stats")
        ax_stats.legend(loc="upper right")
        
        # Events timeline subplot
        ax_events.set_xlabel("Time (s)")
        ax_events.set_ylabel("Events")
        ax_events.set_title("Event Timeline")
        ax_events.set_ylim(-0.5, 4.5)
        ax_events.set_yticks([0, 1, 2, 3, 4])
        ax_events.set_yticklabels(["Stable", "Eyes Closed", "Eyes Open", "Low Focus", "High Focus"])
        ax_events.grid(True, alpha=0.3)
        
        self._fig.tight_layout()

    def _update_dashboard(self):
        if self._fig is None:
            return
        if len(self.t_hist) == 0:
            return
        t = list(self.t_hist)
        
        # Update band lines
        for name, line in self._lines_bands.items():
            y = list(self.band_hist[name])
            line.set_data(t, y)
        # Autoscale bands axis
        ax_bands = self._axes[0]
        ax_bands.relim()
        ax_bands.autoscale_view(True, True, True)
        ax_bands.set_ylim(0, 1)

        # Update state text
        self._state_text.set_text(f"State: {self.current_state}")

        # Update stats lines
        ax_stats = self._axes[1]
        for key, line in self._lines_stats.items():
            y = list(self.stats_hist[key])
            line.set_data(t, y)
        ax_stats.relim()
        ax_stats.autoscale_view(True, True, True)

        # Update events timeline
        ax_events = self._axes[2]
        ax_events.clear()
        ax_events.set_ylabel("Events")
        ax_events.set_title("Event Timeline")
        ax_events.set_ylim(-0.5, 4.5)
        ax_events.set_yticks([0, 1, 2, 3, 4])
        ax_events.set_yticklabels(["Stable", "Eyes Closed", "Eyes Open", "Low Focus", "High Focus"])
        ax_events.grid(True, alpha=0.3)
        
        # Plot events as colored dots
        if self.event_times:
            current_time = t[-1] if t else 0
            tmin = max(0.0, current_time - 60.0)
            
            # Filter events within visible time window
            visible_events = [(event_time, event_name) for event_time, event_name in self.event_times 
                            if event_time >= tmin and event_time <= current_time]
            
            if visible_events:
                event_colors = {
                    "Eyes Closed Event": "blue",
                    "Eyes Open Event": "green",
                    "Low Focus Event": "red",
                    "High Focus Event": "orange"
                }
                
                event_y_pos = {
                    "Eyes Closed Event": 1,
                    "Eyes Open Event": 2,
                    "Low Focus Event": 3,
                    "High Focus Event": 4
                }
                
                for event_time, event_name in visible_events:
                    y_pos = event_y_pos.get(event_name, 0)
                    color = event_colors.get(event_name, "gray")
                    ax_events.scatter(event_time, y_pos, c=color, s=100, alpha=0.8, edgecolors='black')

        # Set x-limits to last 60s
        tmin = max(0.0, t[-1] - 60.0)
        for ax in self._axes:
            ax.set_xlim(tmin, t[-1] + 1e-3)

        # Use non-blocking canvas update
        try:
            self._fig.canvas.flush_events()
        except:
            pass

    async def dashboard_loop(self):
        self._init_dashboard()
        while self._running:
            try:
                await asyncio.sleep(REPORT_INTERVAL)
                self._update_dashboard()
            except Exception as e:
                logger.warning(f"Dashboard update error: {e}")
                # Continue running even if dashboard update fails
                continue

async def main():
    print("EEG BLE Data Collector")
    print("======================")
    print("Make sure your ESP32C3 with BioAmp EXG Pill is running and advertising as BLE server\n")

    collector = EEGDataCollector()

    # Scan
    logger.info("Scanning for device named '%s'...", DEVICE_NAME)
    devices = await BleakScanner.discover(timeout=SCAN_TIMEOUT)
    target = next((d for d in devices if d.name and DEVICE_NAME in d.name), None)

    if not target:
        logger.error("Device not found. Is it advertising?")
        return

    logger.info("Found device: %s (%s)", target.name, target.address)

    # Connect
    async with BleakClient(target.address) as client:
        if not client.is_connected:
            logger.error("Failed to connect.")
            return

        logger.info("Connected successfully!")

        # Subscribe to EEG notifications
        await client.start_notify(CHARACTERISTIC_UUID, collector.handle_notification)

        print("Receiving EEG data... Press Ctrl+C to stop.")
        # Start periodic reporting task
        report_task = asyncio.create_task(collector.periodic_report())
        # Start dashboard loop
        plot_task = asyncio.create_task(collector.dashboard_loop())

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            
            # Save and print final event counts
            summary_filename = f"event_summary_{collector.start_time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(summary_filename, "w") as f:
                f.write("--- Event Summary ---\n")
                for event, count in collector.event_counts.items():
                    f.write(f"{event}: {count}\n")
                f.write("---------------------\n")
            logger.info(f"Event summary saved to {summary_filename}")

            print("\n--- Event Summary ---")
            for event, count in collector.event_counts.items():
                print(f"{event}: {count}")
            print("---------------------\n")

            await client.stop_notify(CHARACTERISTIC_UUID)
            collector._running = False
            try:
                await asyncio.wait_for(report_task, timeout=2.0)
            except asyncio.TimeoutError:
                report_task.cancel()
            try:
                await asyncio.wait_for(plot_task, timeout=2.0)
            except asyncio.TimeoutError:
                plot_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
