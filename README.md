# NEOAGF Headband (WIP)

A work-in-progress repository for the NEOAGF headband: a tACS stimulator and EEG reader. The repo currently contains:

- ESP32 firmware to acquire EEG via ADS1015 and stream filtered data over BLE
- A serial-controlled signal generator for a GP8413 DAC (for stimulation signal prototyping)
- A Python BLE client for live visualization, band-power analysis, and CSV logging
- A small helper script to design Butterworth filters

Use at your own risk. See Safety section below.

## Repository Structure

- `esp32/neoagf_bt_eeg_reader.ino` — ESP32C3 BLE server that samples EEG from ADS1015, filters it in real time (DC block, 50/60 Hz notch, 45 Hz LP), and notifies a BLE characteristic.
- `esp32/neoagf_stimulation.ino` — GP8413 DAC signal generator with a serial command interface for frequency/amplitude presets in EEG bands (for tACS prototyping). 
- `eeg_ble_client_fixed.py` — Python client using Bleak to discover/connect, plot basic dashboards, compute band powers, and log data to CSV.
- `butter-calc.py` — Helper for generating Butterworth coefficients (SciPy).
- `neoagf-venv/` — A Python virtual environment directory (optional, local only).

## Hardware (current focus)

- Acquisition:
  - Seeed XIAO ESP32C3 (ESP32-C3)
  - ADS1015 ADC (I2C, default address 0x48)
  - BioAmp EXG Pill + EEG electrodes
- Stimulation (prototype signal generation):
  - GP8413 (GP8xxx family) DAC module
  - I2C wiring per board; code currently configures `Wire.setPins(7, 8)`

Notes:
- The BLE firmware explicitly targets XIAO ESP32C3 and initializes BLE as `ESP32_EEG`.
- The stimulation sketch’s header mentions Seeeduino Nano + Grove I2C; adapt wiring/board selection as needed. The code uses the DFRobot GP8xxx library and a 15-bit DAC range.

## Firmware: EEG over BLE

File: `esp32/neoagf_bt_eeg_reader.ino`

Features:
- SAMPLE_RATE = 250 Hz
- Filters: DC blocker (~0.5 Hz), notch at 50/60 Hz, 4th-order low-pass at 45 Hz
- BLE service/characteristic UUIDs: 
  - Service: `4fafc201-1fb5-459e-8fcc-c5c9c331914b`
  - Characteristic: `beb5483e-36e1-4688-b7f5-ea07361b26a8`
- Sends filtered EEG voltage as text floats via notifications

Required Arduino libraries:
- ESP32 Arduino core
- Adafruit ADS1X15
- Arduino BLE (ESP32 core includes BLE support; this sketch uses `BLEDevice`, `BLEServer`, `BLECharacteristic` APIs)

Wiring (typical XIAO ESP32C3):
- I2C (SDA/SCL) to ADS1015
- BioAmp EXG Pill output to ADS1015 AIN3 (sketch reads channel 3)
- Status LED on `LED_PIN` (5)

Build & Flash (Arduino IDE):
1. Select the correct board (XIAO ESP32C3) and port.
2. Install required libraries.
3. Open `esp32/neoagf_bt_eeg_reader.ino` and upload.
4. The device should advertise as `ESP32_EEG`.

## Firmware: GP8413 DAC Signal Generator (tACS prototyping)

File: `esp32/neoagf_stimulation.ino`

Features:
- Generates a sine centered at 2.5 V with adjustable amplitude and frequency
- Default sample rate 2 kHz, 15-bit DAC scaling
- Presets for EEG bands stored in PROGMEM: Delta 2 Hz, Theta 6 Hz, Alpha 10 Hz, Beta 20 Hz, Gamma 40 Hz
- Serial command interface (115200 baud):
  - `0`..`4` — select band (2, 6, 10, 20, 40 Hz)
  - `f<val>` — set frequency in Hz (0.1–100)
  - `a<val>` — set amplitude in mV (0.1–100.0), centered around 2.5 V
  - `+` / `-` — increase/decrease amplitude (bounded)
  - `help` — show help
  - `test` — run simple DAC step test

Required Arduino libraries:
- Wire
- DFRobot_GP8XXX

Wiring:
- I2C to the GP8413 module; code sets I2C pins via `Wire.setPins(7, 8)` and uses device address `0x59`

Build & Run:
1. Open serial monitor at 115200.
2. Use commands above to shape the waveform.

Important: This is for bench/prototyping. Any use for human stimulation must include proper isolation, current limiting, medical-grade safety considerations, and review by qualified professionals.

## Python BLE Client

File: `eeg_ble_client_fixed.py`

What it does:
- Discovers the device (looks for names containing `ESP32_EE`, which matches `ESP32_EEG`)
- Subscribes to notifications and logs values to `eeg_data_YYYYMMDD_HHMMSS.csv`
- Computes relative band powers (delta/theta/alpha/beta/gamma < 100 Hz)
- Displays a simple live dashboard (band powers and basic signal stats)

Dependencies (Python 3.10+ recommended):
- bleak
- numpy
- scipy
- matplotlib

Quick start:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install bleak numpy scipy matplotlib
python eeg_ble_client_fixed.py
```

If you already have `neoagf-venv/`, you can use it instead; ensure packages above are installed.

## Helper: Filter Coefficients

File: `butter-calc.py`

- Uses SciPy to compute a 4th-order bandpass (0.5–90 Hz overall) at fs=250 Hz and prints `b, a`.

Run:
```bash
python butter-calc.py
```

## Safety

This project involves EEG acquisition and transcranial electrical stimulation concepts. Incorrect use can be dangerous.
- Do not apply stimulation to humans unless you fully understand the risks and comply with applicable laws/regulations.
- Always use proper isolation, current limiting, and medically appropriate hardware if experimenting beyond bench tests.
- Data and code are provided as-is, without any warranty or claim of fitness for medical use.

## Roadmap (provisional)
- [ ] Hardware integration and enclosure for the headband
- [ ] Validate stimulation current paths and safety circuits
- [ ] Improve BLE data format (binary float packets, sequence numbers)
- [ ] Expand client visualization (PSDs, spectrograms, markers)
- [ ] End-to-end calibration and documentation

## Contributing
PRs and issues are welcome. Please describe hardware, firmware, and environment when reporting problems.

## License
Specify a license in this repository (e.g., MIT/Apache-2.0). If none is provided yet, all rights reserved by default.
