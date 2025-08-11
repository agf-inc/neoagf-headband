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
DEVICE_NAME = "ESP32_EE"  # Match the name your ESP32 is advertising
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
SCAN_TIMEOUT = 10.0
SAMPLE_RATE_HZ = 250  # EEG sample rate
WINDOW_SECONDS = 5     # Sliding window length for analysis
REPORT_INTERVAL = 1.0  # Seconds between reports

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        # Matplotlib objects
        self._fig = None
        self._axes = None
        self._lines_bands = {}
        self._lines_stats = {}

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
            n = data.size
            duration = n / SAMPLE_RATE_HZ
            stats = self._compute_signal_stats(data)
            bands = self._compute_band_powers(data)
            new_samples = self.data_count - last_count
            last_count = self.data_count

            logger.info(
                (
                    "Stats over last %.1fs (%d samples, +%d): "
                    "Vmin=%.6f V, Vmax=%.6f V, Vavg=%.6f V, Vrms=%.6f V | "
                    "Bands (rel power): delta=%.2f, theta=%.2f, alpha=%.2f, beta=%.2f, gamma=%.2f"
                ),
                duration,
                n,
                new_samples,
                stats["vmin"],
                stats["vmax"],
                stats["vavg"],
                stats["vrms"],
                bands.get("delta", float('nan')),
                bands.get("theta", float('nan')),
                bands.get("alpha", float('nan')),
                bands.get("beta", float('nan')),
                bands.get("gamma", float('nan')),
            )

            # Append to histories (time in seconds since start)
            t = (datetime.now() - self.start_time).total_seconds()
            self.t_hist.append(t)
            for name in self.bands:
                self.band_hist[name].append(bands.get(name, float('nan')))
            for key in self.stats_hist:
                self.stats_hist[key].append(stats.get(key, float('nan')))

    def _init_dashboard(self):
        if self._fig is not None:
            return
        plt.ion()
        self._fig, self._axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax_bands, ax_stats = self._axes

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

        # Stats subplot
        for key, color in zip(["vmin", "vmax", "vavg", "vrms"], ["C0", "C3", "C2", "C1"]):
            (line,) = ax_stats.plot([], [], label=key, color=color)
            self._lines_stats[key] = line
        ax_stats.set_xlabel("Time (s)")
        ax_stats.set_ylabel("Voltage (V)")
        ax_stats.set_title("Signal Stats")
        ax_stats.legend(loc="upper right")
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

        # Update stats lines
        ax_stats = self._axes[1]
        for key, line in self._lines_stats.items():
            y = list(self.stats_hist[key])
            line.set_data(t, y)
        ax_stats.relim()
        ax_stats.autoscale_view(True, True, True)

        # Set x-limits to last 60s
        tmin = max(0.0, t[-1] - 60.0)
        for ax in self._axes:
            ax.set_xlim(tmin, t[-1] + 1e-3)

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    async def dashboard_loop(self):
        self._init_dashboard()
        while self._running:
            await asyncio.sleep(REPORT_INTERVAL)
            self._update_dashboard()

async def main():
    print("EEG BLE Data Collector")
    print("======================")
    print("Make sure your ESP32C3 with BioAmp EXG Pill is running and advertising as BLE server\n")

    collector = EEGDataCollector()

    # Scan
    logger.info("Scanning for device named '%s'...", DEVICE_NAME)
    devices = await BleakScanner.discover(timeout=SCAN_TIMEOUT)
    for d in devices:
        logger.info("BLE DEVICE: %s", d.name)
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
