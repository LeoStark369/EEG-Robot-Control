"""
bci_robot_control.py
=====================
A BrainFlow-based Brain-Computer Interface (BCI) that turns EEG signals into
robot control commands using Motor Imagery classification.

HOW IT WORKS (real-world BCI, not sci-fi mind reading):
  1. Stream raw EEG from a BrainFlow-supported headset (OpenBCI, Muse, Neurosity, etc.)
  2. Band-pass filter + extract band-power features (mu 8-12Hz, beta 13-30Hz)
  3. User "trains" the system by imagining LEFT hand, RIGHT hand, and FEET
     movement while labeled data is collected (a short calibration wizard)
  4. An LDA classifier learns to tell these patterns apart
  5. Live EEG windows are classified in real time -> mapped to robot commands
     -> sent over serial to a robot (Arduino, ESP32, ROS bridge, etc.)

REQUIREMENTS:
  pip install brainflow numpy scipy scikit-learn pyserial

HARDWARE:
  - Any BrainFlow-supported EEG board (set BOARD_ID / params below)
  - A robot that accepts single-character serial commands, e.g. an Arduino
    sketch that reads 'F','B','L','R','S' from Serial and drives motors.
    (Swap RobotInterface.send() for WiFi/ROS/BLE if your robot uses that instead)

USAGE:
  python bci_robot_control.py --board synthetic --serial-port COM5
  (use --board synthetic to test the whole pipeline with no real headset)
"""

import argparse
import time
import sys
import numpy as np
from collections import deque

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

try:
    import serial
except ImportError:
    serial = None


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

COMMANDS = {
    "left_hand": "L",   # -> turn left
    "right_hand": "R",  # -> turn right
    "feet": "F",         # -> move forward
    "rest": "S",          # -> stop
}

CALIBRATION_TRIALS_PER_CLASS = 8
TRIAL_DURATION_SEC = 4
WINDOW_SEC = 2          # live classification window length
STEP_SEC = 0.5          # how often we emit a new command
CONFIDENCE_THRESHOLD = 0.55  # below this, we send STOP for safety


# --------------------------------------------------------------------------
# ROBOT INTERFACE
# --------------------------------------------------------------------------

class RobotInterface:
    """Sends simple single-char commands to a robot over serial.
    Replace this class internals if your robot uses WiFi/ROS/BLE instead."""

    def __init__(self, port: str = None, baud: int = 115200, dry_run: bool = False):
        self.dry_run = dry_run or serial is None or port is None
        self.ser = None
        if not self.dry_run:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # allow Arduino auto-reset to settle
        self._last_cmd = None

    def send(self, cmd: str):
        if cmd == self._last_cmd:
            return  # avoid spamming identical commands
        self._last_cmd = cmd
        if self.dry_run:
            print(f"[ROBOT] (dry-run) -> {cmd}")
        else:
            self.ser.write(cmd.encode("utf-8"))

    def stop(self):
        self.send("S")

    def close(self):
        self.stop()
        if self.ser:
            self.ser.close()


# --------------------------------------------------------------------------
# SIGNAL PROCESSING
# --------------------------------------------------------------------------

class FeatureExtractor:
    """Turns a raw EEG window into a feature vector using band-power."""

    BANDS = {
        "theta": (4, 8),
        "mu": (8, 12),
        "beta": (13, 30),
    }

    def __init__(self, sampling_rate: int, eeg_channels: list):
        self.sr = sampling_rate
        self.eeg_channels = eeg_channels

    def extract(self, data: np.ndarray) -> np.ndarray:
        """data shape: (num_channels, num_samples) — full board data array."""
        feats = []
        for ch in self.eeg_channels:
            sig = data[ch].copy()
            DataFilter.detrend(sig, DetrendOperations.LINEAR.value)
            DataFilter.perform_bandpass(
                sig, self.sr, 3.0, 45.0, 4,
                FilterTypes.BUTTERWORTH.value, 0
            )
            for lo, hi in self.BANDS.values():
                band_sig = sig.copy()
                DataFilter.perform_bandpass(
                    band_sig, self.sr, (lo + hi) / 2, hi - lo, 4,
                    FilterTypes.BUTTERWORTH.value, 0
                )
                power = np.mean(band_sig ** 2)
                feats.append(power)
        return np.array(feats)


# --------------------------------------------------------------------------
# MAIN BCI CONTROLLER
# --------------------------------------------------------------------------

class BCIRobotController:
    def __init__(self, board_id: int, params: BrainFlowInputParams, robot: RobotInterface):
        self.board = BoardShim(board_id, params)
        self.sr = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.fe = FeatureExtractor(self.sr, self.eeg_channels)
        self.robot = robot
        self.clf = LinearDiscriminantAnalysis()
        self.scaler = StandardScaler()
        self.labels = list(COMMANDS.keys())

    # ---------------- connection ----------------
    def connect(self):
        self.board.prepare_session()
        self.board.start_stream()
        print("EEG stream started.")

    def disconnect(self):
        self.board.stop_stream()
        self.board.release_session()
        print("EEG stream stopped.")

    # ---------------- calibration wizard ----------------
    def calibrate(self):
        """Guides the user through imagining each mental command and
        collects labeled training data. Very easy: just follow the prompts."""
        print("\n=== CALIBRATION ===")
        print("You'll be prompted to imagine an action several times.")
        print("Just relax and picture the movement vividly — no need to move.\n")

        X, y = [], []
        trial_plan = self.labels * CALIBRATION_TRIALS_PER_CLASS
        np.random.shuffle(trial_plan)

        for i, label in enumerate(trial_plan):
            input(f"[{i+1}/{len(trial_plan)}] Get ready to imagine: '{label.upper()}' "
                  f"— press Enter to start ({TRIAL_DURATION_SEC}s)...")
            print("  >> GO <<")
            self.board.get_board_data()  # clear buffer
            time.sleep(TRIAL_DURATION_SEC)
            data = self.board.get_board_data()
            if data.shape[1] < self.sr:  # not enough samples, skip
                print("  (not enough data, skipping trial)")
                continue
            feats = self.fe.extract(data)
            X.append(feats)
            y.append(label)
            print("  done.\n")

        X = np.array(X)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        acc = self.clf.score(Xs, y)
        print(f"Calibration complete. Training accuracy: {acc*100:.1f}%\n")

    # ---------------- live control loop ----------------
    def run_live(self, duration_sec: float = None):
        print("=== LIVE CONTROL === (Ctrl+C to stop)\n")
        window_samples = int(WINDOW_SEC * self.sr)
        start_time = time.time()
        try:
            while True:
                time.sleep(STEP_SEC)
                data = self.board.get_current_board_data(window_samples)
                if data.shape[1] < window_samples * 0.8:
                    continue  # not enough data yet

                feats = self.fe.extract(data).reshape(1, -1)
                feats_s = self.scaler.transform(feats)
                probs = self.clf.predict_proba(feats_s)[0]
                best_idx = int(np.argmax(probs))
                confidence = probs[best_idx]
                predicted_label = self.clf.classes_[best_idx]

                if confidence < CONFIDENCE_THRESHOLD:
                    cmd = COMMANDS["rest"]
                    print(f"  low confidence ({confidence:.2f}) -> STOP")
                else:
                    cmd = COMMANDS[predicted_label]
                    print(f"  {predicted_label:10s} (conf {confidence:.2f}) -> {cmd}")

                self.robot.send(cmd)

                if duration_sec and (time.time() - start_time) > duration_sec:
                    break
        except KeyboardInterrupt:
            print("\nStopping by user request.")
        finally:
            self.robot.stop()


# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------

def build_board_params(args) -> (int, BrainFlowInputParams):
    params = BrainFlowInputParams()
    board_map = {
        "synthetic": BoardIds.SYNTHETIC_BOARD,
        "cyton": BoardIds.CYTON_BOARD,
        "cyton_daisy": BoardIds.CYTON_DAISY_BOARD,
        "muse2": BoardIds.MUSE_2_BOARD,
        "ganglion": BoardIds.GANGLION_BOARD,
    }
    if args.board not in board_map:
        print(f"Unknown board '{args.board}'. Options: {list(board_map.keys())}")
        sys.exit(1)
    board_id = board_map[args.board].value
    if args.eeg_serial_port:
        params.serial_port = args.eeg_serial_port
    if args.mac_address:
        params.mac_address = args.mac_address
    return board_id, params


def main():
    parser = argparse.ArgumentParser(description="BrainFlow EEG -> Robot control")
    parser.add_argument("--board", default="synthetic",
                         help="synthetic | cyton | cyton_daisy | muse2 | ganglion")
    parser.add_argument("--eeg-serial-port", default=None,
                         help="Serial port for the EEG headset (e.g. COM4 or /dev/ttyUSB0)")
    parser.add_argument("--mac-address", default=None,
                         help="MAC address for BLE boards like Muse")
    parser.add_argument("--serial-port", default=None,
                         help="Serial port for the ROBOT (e.g. COM5). Omit for dry-run/testing.")
    parser.add_argument("--skip-calibration", action="store_true",
                         help="Skip calibration (only useful for quick pipeline testing)")
    args = parser.parse_args()

    board_id, params = build_board_params(args)
    robot = RobotInterface(port=args.serial_port, dry_run=args.serial_port is None)
    controller = BCIRobotController(board_id, params, robot)

    controller.connect()
    try:
        if not args.skip_calibration:
            controller.calibrate()
        else:
            print("Skipping calibration — classifier untrained, using dummy pass-through.\n")
        controller.run_live()
    finally:
        controller.disconnect()
        robot.close()


if __name__ == "__main__":
    main()
