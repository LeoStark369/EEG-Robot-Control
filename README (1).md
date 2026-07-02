# 🧠 BCI Robot Control

Control a robot in real time using **EEG brain signals** — no buttons, no joystick, just imagined movement. Built with [BrainFlow](https://brainflow.org/), a real-time motor-imagery classifier, and a simple serial interface to drive any robot.

> **Developer:** Leo Hawking

---

## ⚡ What This Is

This project is a **Brain-Computer Interface (BCI)** that turns raw EEG signals into robot movement commands. It is *not* literal mind-reading — it works by detecting a well-documented neurological phenomenon called **Motor Imagery**: the brain produces measurable, repeatable patterns when a person *imagines* moving a limb, even without physically moving it.

| Imagined Action     | Robot Command |
|----------------------|---------------|
| Left hand movement   | Turn Left     |
| Right hand movement  | Turn Right    |
| Feet movement        | Move Forward  |
| Relaxed / unclear     | Stop (safety default) |

---

## 🛠️ Libraries & Tech Used

| Library | Purpose |
|---|---|
| [`brainflow`](https://brainflow.org/) | EEG headset connection, streaming, and signal filtering |
| [`numpy`](https://numpy.org/) | Numerical operations on EEG arrays |
| [`scipy`](https://scipy.org/) *(via BrainFlow filters)* | Signal processing backend |
| [`scikit-learn`](https://scikit-learn.org/) | LDA (Linear Discriminant Analysis) classifier for real-time prediction |
| [`pyserial`](https://pyserial.readthedocs.io/) | Sends movement commands to the robot over serial (Arduino/ESP32/etc.) |

**Supported EEG hardware:** any BrainFlow-supported board — OpenBCI Cyton, Cyton+Daisy, Ganglion, Muse 2, and more. A `synthetic` virtual board is included so you can test the entire pipeline with **zero hardware**.

---

## 🧩 How It's Built (Simple Explanation)

Think of it like teaching a new "language" to a computer — the language of *your* brain.

1. **Listen** 🎧 — EEG electrodes pick up tiny electrical signals from the brain via BrainFlow.
2. **Clean** 🧹 — Raw brain signals are noisy, so filters strip out drift and irrelevant frequencies, keeping only the useful 3–45 Hz range.
3. **Translate** 🔢 — The cleaned signal is converted into a short list of numbers representing brainwave "power" in specific bands — this is the *feature vector*.
4. **Teach (Calibration)** 🎓 — The user imagines Left, Right, and Feet movement a few times each while the system labels and records the pattern. A machine learning model (LDA) learns to recognize each pattern.
5. **Predict Live** 🔮 — Once trained, the system continuously reads new brain data, compares it to what it learned, and predicts what the user is imagining — several times per second.
6. **Act** 🤖 — The predicted action is sent as a simple command (`F`, `L`, `R`, `S`) over serial to the robot's controller (e.g. an Arduino), which drives the motors.

If the model isn't confident about the prediction, it defaults to **Stop** instead of guessing — a basic but important safety behavior for anything physically moving.

---

## 🧠 The Neuroscience Behind It

### Motor Imagery
When you physically move your hand, your brain's motor cortex generates electrical activity to control the muscles. Interestingly, when you **imagine** the same movement without doing it, the motor cortex produces a *similar, measurable* pattern — just weaker. This is called **Motor Imagery (MI)**, and it's one of the most well-studied, reproducible signals in BCI research.

### The Brainwave Bands That Matter
EEG signals are commonly split into frequency bands. This project focuses on two that are directly tied to motor activity:

- **Mu rhythm (8–12 Hz):** Present over the motor cortex at rest; it *suppresses* (drops in power) when a person moves or imagines moving. This drop is called **Event-Related Desynchronization (ERD)**.
- **Beta rhythm (13–30 Hz):** Similarly linked to motor planning and imagined movement, also showing ERD patterns during motor imagery, and a rebound (**Event-Related Synchronization**) shortly after.
- **Theta (4–8 Hz):** Included as a supporting feature; often reflects general cognitive engagement/focus.

### Why Calibration Is Necessary
Every brain has a slightly different signal "fingerprint" — electrode placement, skull thickness, and individual neurology all affect signal strength and shape. That's why this system trains a fresh classifier for each user rather than using one fixed model for everyone. It's the same reason voice assistants often ask you to "train" them to your voice.

### Realistic Expectations
This is real, published BCI science — not science fiction. With a cooperative, trained user, typical accuracy for 2–3 imagined actions is around **70–85%**. It's closer to predictive text than telepathy: reliable enough to be useful, but not perfect, which is why the low-confidence "Stop" fallback matters for anything controlling a physical robot.

---

## 📦 Installation

```bash
pip install brainflow numpy scipy scikit-learn pyserial
```

## 🚀 Usage

**Test the full pipeline with no hardware (synthetic EEG):**
```bash
python bci_robot_control.py --board synthetic
```

**Run with real EEG hardware + a real robot:**
```bash
python bci_robot_control.py --board cyton --eeg-serial-port COM4 --serial-port COM5
```

### CLI Options
| Flag | Description |
|---|---|
| `--board` | EEG board type: `synthetic`, `cyton`, `cyton_daisy`, `muse2`, `ganglion` |
| `--eeg-serial-port` | Serial port of the EEG headset (e.g. `COM4`, `/dev/ttyUSB0`) |
| `--mac-address` | MAC address for BLE boards like Muse |
| `--serial-port` | Serial port of the robot's controller (omit for dry-run/testing) |
| `--skip-calibration` | Skips training (for quick pipeline debugging only) |

---

## 🤖 Robot Side (Arduino Example)

The robot just needs to listen for single characters over serial:

```cpp
void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    switch (cmd) {
      case 'F': moveForward(); break;
      case 'L': turnLeft();    break;
      case 'R': turnRight();   break;
      case 'S': stopMotors();  break;
    }
  }
}
```

Swap this out for WiFi, Bluetooth, or ROS if your robot uses a different control scheme.

---

## ⚠️ Safety Notes

- Always test with `--board synthetic` and `dry-run` mode (no `--serial-port`) before connecting a real robot.
- The confidence threshold defaults to sending **Stop** when the system isn't sure — don't lower this without testing thoroughly.
- Keep a physical kill switch or manual override on any real robot controlled this way.

---

## 📄 License

MIT — free to use, modify, and build on.

---

## 👤 Developer

**Leo Hawking**
