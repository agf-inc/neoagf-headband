# neoagf BLE Connectivity Guide (Client Implementation)

This document describes how to connect to and control the ESP32-based EEG + tDCS device from a BLE client. It captures the GATT layout, UUIDs, message formats, and example sequences corresponding to `esp32/neoagf_eeg_tdcs_combo.ino`.

## Overview

- Device Name: `NEOAGF`
- Transport: BLE GATT
- Security: No bonding/pairing required (open access)
- Advertising:
  - Primary Service UUID: `f47ac10b-58cc-4372-a567-0e02b2c3d479`

## GATT Layout

- Service UUID: `f47ac10b-58cc-4372-a567-0e02b2c3d479`
  - Characteristic: EEG Data
    - UUID: `f47ac10b-58cc-4372-a567-0e02b2c3d480`
    - Properties: `READ`, `NOTIFY`, `INDICATE`
    - CCCD: Enable notifications to receive streaming data
    - Payload: ASCII text of a float value in volts (see Data Formats)
  - Characteristic: Control
    - UUID: `f47ac10b-58cc-4372-a567-0e02b2c3d481`
    - Properties: `READ`, `WRITE`, `NOTIFY`
    - CCCD: Enable notifications to receive command responses and async status
    - Payload: ASCII commands and ASCII responses

Notes:
- The control characteristic both accepts writes and emits responses via notifications on the same characteristic.
- Keep writes small (<= 20 bytes) to be compatible with default MTU on most stacks. The firmware does not rely on increased MTU.

## Data Formats

### EEG Data (Notifications)
- Topic: EEG characteristic `f47ac10b-58cc-4372-a567-0e02b2c3d480`
- Rate: ~250 samples/second (250 Hz)
- Format: ASCII string of a single float with 6 decimal places
- Units: Volts
- Example payloads:
  - `0.001234\n` (newline not guaranteed; treat as plain string)
  - `-0.000567`

### Control (Commands and Responses)
- Topic: Control characteristic `f47ac10b-58cc-4372-a567-0e02b2c3d481`
- Client actions: Write ASCII command (Write With Response recommended)
- Device responses: Short ASCII strings via NOTIFY on the same characteristic
- Trimming: Whitespace is trimmed; commands are case-insensitive
- Read behavior: Reading this characteristic returns a STATUS snapshot JSON string

Supported commands:
- `MODE EEG`
  - Switches to EEG-only mode (tDCS ramps to 0 mA)
  - Response: `OK MODE EEG`
- `MODE STIM`
  - Enables stimulation mode (tDCS ramps to target current)
  - Response: `OK MODE STIM`
- `MODE NO_OP`
  - Neutral mode; EEG sampling/stimulation idle. Target current forced to 0.
  - Response: `OK MODE NO_OP`
- `I+`
  - Increase tDCS target current by the configured step (default 0.1 mA)
  - Response: `OK I=<value>` (one decimal)
- `I-`
  - Decrease tDCS target current by the configured step
  - Response: `OK I=<value>`
- `I=<mA>`
  - Set absolute target current in mA, clamped to `[0,25]`
  - Example: `I=2.5` => `OK I=2.5`
- `STEP=<mA>`
  - Set increment step for `I+`/`I-` (range `[0.1, 5.0]` mA)
  - Response: `OK STEP=<value>` (two decimals)
- `STATUS?`
  - Query current status. Response is a JSON-like string:
  - Example: `{"bt":"connected","mode":"STIM","I":1.20,"target":2.00}`

Error responses:
- `ERR MODE?` (invalid mode value in `MODE` command)
- `ERR UNKNOWN` (command not recognized)

Notes:
- Default mode after boot is `NO_OP`. EEG sampling and notifications start only after `MODE EEG`.
- tDCS current is applied gradually (ramped) at 0.1 mA/s by default. The `I` value in `STATUS?` reflects the current output; `target` is the goal.
- In `EEG` mode, the firmware forces `target=0.0` mA and ramps down if needed.

## Client Connection Flow

1. Scan for devices:
   - Filter by device name `NEOAGF` or by Service UUID `f47ac10b-58cc-4372-a567-0e02b2c3d479`.
2. Connect.
3. Discover services and characteristics.
4. Enable notifications:
   - EEG characteristic `f47ac10b-58cc-4372-a567-0e02b2c3d480`
   - Control characteristic `f47ac10b-58cc-4372-a567-0e02b2c3d481`
5. (Optional) Write control commands to configure mode/current.

## Example Sequences

- Start EEG streaming:
  - Subscribe to EEG notifications.
  - Write `MODE EEG` → expect `OK MODE EEG`.
  - You will receive ~250 messages/s with ASCII floats.

- Enable stimulation at 2 mA:
  - Write `MODE STIM` to Control → expect `OK MODE STIM`
  - Write `I=2` to Control → expect `OK I=2.0`
  - Optionally `STATUS?` → `{"bt":"connected","mode":"STIM","I":<ramping>,"target":2.00}`

- Adjust step and nudge current:
  - `STEP=0.5` → `OK STEP=0.50`
  - `I+` → `OK I=<new>`

- Return to EEG only:
  - `MODE EEG` → `OK MODE EEG` (device ramps output to 0 mA)
  - Optionally `STATUS?` → `{"bt":"connected","mode":"EEG","I":0.00,"target":0.00}`

## Platform Notes

- iOS/Android/Desktop BLE stacks differ slightly. Recommended:
  - Use Write With Response for commands to ensure reliability.
  - Subscribe to Control notifications to receive responses and async status updates.
  - Handle notification rate on EEG data (~250 Hz). If your stack queues notifications, process efficiently or throttle by unsubscribing when not needed.

## Reference

- Firmware file: `esp32/neoagf_eeg_tdcs_combo.ino`
- ADS1015 channel used for EEG: AIN3 (implementation detail; client does not need to know)
- tDCS driver: DFRobot GP8302
