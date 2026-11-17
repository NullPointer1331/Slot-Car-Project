# Slot-Car-Project

Installation Instructions
Run the following commands in a terminal pointing towards the Slot-Car-Project folder
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MicroPython data-collection deployment
1. Flash MicroPython onto the ESP32 and copy `DataCollection/code.py` to the
   board as `code.py`.
2. Copy the `Common` folder to the board. In `Common/pins.py`, leave
   `SIMULATION = False` for the Feather V2 or set it to `True` to select the
   Wokwi DevKitC pin assignments. The MicroPython port uses GPIO numbers directly
   (`machine.Pin`, `machine.PWM`, `machine.ADC`, `machine.I2C`, and `machine.SPI`)
   instead of the CircuitPython `board` and `digitalio` modules.
3. Copy an SDHC-compatible MicroPython SPI SD-card driver `sdcard.py` to the
   board (or ensure it is provided by the firmware). Some driver versions
   incorrectly send CMD16 to SDHC cards; only byte-addressed SDSC cards need
   that command. The logger mounts the card at `/sd` before opening `data.csv`.
4. Insert a FAT-formatted microSD card. Data collection writes `/sd/data.csv`;
   the controller starts when track power is detected. Each row contains sensor
   features, distance since the last landmark, car state, requested throttle,
   the safety-limited throttle command, and safety cap. Track-specific sections
   and lap length are intentionally not assumed; calibrate the actual track
   before adding track geometry to a control model.

The host Python dependencies in `requirements.txt` are for model development;
they are separate from the MicroPython firmware running on the ESP32.

## Training a throttle model

With data collection laps in `Training/data.csv`, run:

```powershell
python Training/main.py Training/data.csv --export Training/throttle_model.json --keras-output Training/throttle.keras
```

The pipeline labels crashes within the next 0.5 seconds, builds a safe target
throttle with a 10% margin, splits complete laps for a 20% test holdout, trains
the 16/16 ReLU model, and exports `/Training/throttle_model.json`.

Copy `Training/throttle_model.json` to the ESP32 as `/throttle_model.json`,
then copy `Execution/code.py` to the board as `code.py`. The execution loop runs
at 50 Hz and uses the safety controller for every motor command. If the model
or its sensor inputs are unavailable, it requests the safe fixed fallback
throttle instead.