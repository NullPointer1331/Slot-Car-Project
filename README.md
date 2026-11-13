# Slot-Car-Project

Installation Instructions
Run the following commands in a terminal pointing towards the Slot-Car-Project folder
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CircuitPython data-collection deployment
1. Flash CircuitPython onto the ESP32 and copy `DataCollection/code.py` to the
   board as `code.py`.
2. Copy the `Common` folder to the board. In `Common/pins.py`, leave
   `SIMULATION = False` for the Feather V2 or set it to `True` to select the
   Wokwi DevKitC pin assignments. The Wokwi MicroPython mockup is not a
   drop-in CircuitPython data-collection setup; CircuitPython still requires
   the physical-compatible ICM-20948, MAX17048, and VL53L4CD drivers/sensors.
3. Install the CircuitPython libraries `adafruit_icm20x`,
   `adafruit_max1704x`, `adafruit_vl53l4cd`, and `adafruit_sdcard`, plus their
   dependencies, from the Adafruit CircuitPython bundle. The firmware must
   provide `countio.Counter` for wheel-encoder sampling.
4. Insert a FAT-formatted microSD card. Data collection writes
   `/sd/data.csv`; the controller starts when track power is detected. Each
   row contains sensor features, distance since the last landmark, car state,
   requested throttle, the safety-limited throttle command, and safety cap.
   Track-specific sections and lap length are intentionally not assumed;
   calibrate the actual track before adding track geometry to a control model.

The host Python dependencies above are for model development and are separate
from the libraries installed on the CircuitPython board.