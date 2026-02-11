## AI-Powered Packaging Inspection System (YOLOv8-OBB)

An industrial-grade automated Quality Control (QC) system designed to detect and verify electronic component packaging using high-speed computer vision. The system integrates **YOLOv8-OBB** for precision slot localization and a custom **geometric analytic algorithm** to ensure all components are placed correctly according to strict assembly rules.

## 📌 Project Overview
This project automates the verification of multi-stage packaging by synchronizing 4 independent camera streams into a unified monitoring grid.
* **AI Architecture:** Dual-model system featuring YOLOv8-OBB for rotated slot detection and standard YOLOv8 for component identification.
* **Logic Engine:** Python-based multi-threading for real-time RTSP processing and geometric verification.
* **Key Innovation:** Implementing a **Collinearity Algorithm** using triangle area calculations to identify 5 specific slots in a tilted or rotated tray.



## 🌟 Key Features
1.  **Oriented Slot Identification Logic:**
    * Directly evaluates **10 combinations** ($\binom{5}{3}$) of detected points to find the primary row of 3 collinear slots.
    * Computes the **Triangle Area** ($S = \frac{1}{2} |x_1(y_2 - y_3) + x_2(y_3 - y_1) + x_3(y_1 - y_2)|$) to verify alignment with microsecond latency.
    * Dynamically maps Local IDs (1-5) to Global IDs (1-10) depending on the camera's designated assembly stage.

2.  **Industrial Process & Stability Management:**
    * **Stability Timer:** Uses a non-blocking timer to ensure components are only marked as "Saved" if they remain valid for **3.0 consecutive seconds**.
    * **Cumulative Allowed Items:** Implements a stage-aware logic where cameras at later stages can recognize items from previous steps but flag "future" components as process violations.
    * **Priority Status Messaging:** Displays real-time alerts for "WRONG ITEM", "MISSING", or "HOLDING" states based on the current assembly flow.

3.  **High-Performance Vision Pipeline:**
    * **Multi-Threaded RTSP:** Captures and processes 4 simultaneous camera streams via TCP to eliminate frame lag and buffering.
    * **OBB Precision:** Extracts `xywhr` (center, size, and rotation) data to maintain accurate slot tracking even if the tray is moved or tilted.
    * **Dynamic 2x2 Grid:** Renders all camera feeds into a single synchronized interface with color-coded status overlays.



## 🔌 System Configuration
The system uses 4 IP cameras to monitor different stages of the packaging process defined in `config.py`:

| Camera | Required Components | Global IDs | Status Messages |
| :--- | :--- | :--- | :--- |
| **Cam 1** | Small LED (x3) | 1, 2, 3 | WAITING / MISSING |
| **Cam 2** | Large LED, Main Board | 4, 5 | HOLDING 3s / SAVED |
| **Cam 3** | RGB, Grey/White Cables | 6, 7, 8 | WRONG ITEM |
| **Cam 4** | Component Bag, Charger | 9, 10 | CHECKLIST SAVED |

## 🏗️ Software Architecture (Logic Pipeline)
The system separates AI inference from the geometric identification logic to ensure maximum processing speed.

| Phase | Component | Logic | Function |
| :--- | :--- | :--- | :--- |
| **Inference** | YOLOv8-OBB | `best.pt` | Detects 5 slots and returns oriented coordinates. |
| **Identification**| Geometry Engine | $\binom{5}{3}$ Combinations | Finds 3 collinear points to assign S1, S2, and S3. |
| **Verification** | Logic Engine | `PACKING_RULES` | Validates if the item in a slot matches the `expected_item`. |
| **Stability** | Timer Class | `time.time()` | Tracks 3.0s duration before marking a slot as `is_saved`. |
| **Display** | UI Thread | OpenCV Grid | Renders OBB frames and status messages in a 2x2 grid. |

## 💻 Installation & Usage

### 1. Environment Setup
Create a dedicated environment for the project:
```bash
conda create -n packaging_ai python=3.9
conda activate packaging_ai
pip install ultralytics opencv-python numpy