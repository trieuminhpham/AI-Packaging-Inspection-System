# AI-Packaging-Inspection-System
An automated industrial quality control system using YOLOv8-OBB and geometric algorithms to detect packaging errors in electronic component assembly via a 4-camera RTSP setup.

✨ Key Features

    Dual-Model AI Architecture:

        OBB Model (best.pt): Uses Oriented Bounding Boxes to precisely detect 5 slot locations regardless of the tray's rotation or tilt.

        Standard Model (best_ck.pt): Detects specific electronic components such as LEDs, PCBs, chargers, and cables.

    Geometric 5-Slot Identification:

        Evaluates (35​) combinations (10 cases) to find the primary collinear group of 3 slots.

        Determines the sequence (Slot 1-5) based on the relative position of the 2-slot and 3-slot rows.

    Industrial Process Logic (config.py):

        Stability Timer: Validates an "OK" state only if the correct component remains in the slot for 3.0 consecutive seconds.

        Cumulative Allowed Items: A smart logic that remembers the process; cameras at later stages can recognize items from previous stages but flag "future" items as a process violation.

    Multi-Camera RTSP Integration: Processes 4 simultaneous IP camera streams (Cam 1-4) displayed in a synchronized 2x2 grid.

🛠 Technology Stack

    Language: Python.

    AI Framework: YOLOv8 (Ultralytics).

    Computer Vision: OpenCV.

    Data Processing: NumPy (Vectorized collinearity checks).

    Protocol: RTSP (Real-Time Streaming Protocol) via TCP for high stability.

📐 The Identification Algorithm

The system identifies the primary row by calculating the area of a triangle formed by any 3 detected slot centers using the analytic formula:
S=21​∣x1​(y2​−y3​)+x2​(y3​−y1​)+x3​(y1​−y2​)∣

If S<Threshold, these three points are identified as Slots 1, 2, and 3. The remaining slots are assigned based on their Euclidean distance and the tray's orientation.
⚙️ Process Configuration (config.py)

The system follows strict packaging rules defined for each production stage:
Camera	Required Components	Status Messages
Cam 1	Small LED (x3)	WAITING / MISSING
Cam 2	Large LED, Main Board	HOLDING 3s / SAVED
Cam 3	RGB Cable, Grey/White Cables	WRONG ITEM
Cam 4	Component Bag, Charger	CHECKLIST SAVED
🚀 Installation & Usage
1. Environment Setup
Bash

conda create -n packaging_ai python=3.9
conda activate packaging_ai
pip install ultralytics opencv-python numpy

2. Execution

    Ensure all IP cameras are active on the local network.

    Configure the RTSP URLs in main.py.

    Run the system:

Bash

python main.py