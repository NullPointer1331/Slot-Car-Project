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
Use the Thonny IDE to interact with the files on the ESP32 after flashing it.
1. Flash MicroPython onto the ESP32 using the commands below and copy `DataCollection/main.py` to the board as `main.py`. Specific details in the commands such as the port used may vary. For my machine, our backup ESP32 was listed as COM6
   ```
   esptool --chip esp32 --baud 115200 --port COM7 erase-flash

   esptool --chip esp32  --baud 115200 --port COM7 write_flash -z 0x1000 ESP32_GENERIC-SPIRAM-20260824-v1.29.0.bin
   ```
2. Copy the `Common` folder to the board. In `Common/pins.py`, leave
   `SIMULATION = False` for the Feather V2 or set it to `True` to select the
   Wokwi DevKitC pin assignments.
3. Copy an SDHC-compatible MicroPython SPI SD-card driver `sdcard.py` to the
   board (or ensure it is provided by the firmware). Some driver versions
   incorrectly send CMD16 to SDHC cards; only byte-addressed SDSC cards need
   that command. The logger mounts the card at `/sd` before opening `data.csv`.
4. Insert a FAT-formatted microSD card. Data collection writes `/sd/data.csv`;
   the controller starts when track power is detected. Each row contains turn
   and grade types, sensor features, distance within the current lap, total
   distance, car state,
   the previous applied throttle, requested throttle, safety-limited throttle
   command, safety cap, and IMU-valid flag. Each collection run has its own ID;
   a lap is counted every two landmarks. Failed IMU reads are recorded as
   invalid rather than substituted with zero-motion readings.
5. Mount the IMU with X forward, Y lateral, and Z vertical, and place the car
   stationary on a level track section during startup calibration. Positive
   gyro Z is treated as a left turn. Turn type (straight/left/right) and grade
   (level/uphill/downhill) are inferred independently from gyro and gravity
   direction; they are estimates, not direct observations of track geometry.
   Longitudinal acceleration can bias the grade estimate. The thresholds and
   filtering are in `Common/section_classifier.py` and need verification/tuning
   on the actual car and track before relying on them for control.

The host Python dependencies in `requirements.txt` are for model development;
they are separate from the MicroPython firmware running on the ESP32.

## Training a throttle model

With data collection laps in `Training/data.csv`, run:

```powershell
python Training/main.py Training/data.csv --export Training/throttle_model.json --keras-output Training/throttle.keras
```

The pipeline validates run/lap/landmark consistency, labels crash samples and
the preceding 0.5 seconds, and builds safe-throttle targets from training laps
only. Target actions come from the fastest 25% of crash-free training laps:
for each position and turn/grade combination, the model learns their median
throttle request with a 10% safety margin. Training balances each lap's total
weight and gives those fast laps extra weight, rather than allowing slower,
longer laps to dominate by producing more samples. Whole run/lap groups remain
separate across training, validation, and test partitions. The 16/16 ReLU
model is exported as `/Training/throttle_model.json`; runtime features include
`previous_throttle` and one-hot turn/grade types.
Partial first/last laps are excluded, and at least three complete laps are
required for the three partitions. Training also reports observed complete-lap
times and the fastest crash-free lap as a baseline, and reports excluded
invalid sensor samples. This is elite-lap imitation, not direct counterfactual
lap-time optimization; hardware trials are needed to measure improvements.
At runtime, invalid IMU readings trigger the fixed safe fallback until inputs
recover.

Copy `Training/throttle_model.json` to the ESP32 as `/throttle_model.json`,
then copy `Execution/main.py` to the board as `main.py`. The execution loop runs
at 50 Hz and uses the safety controller for every motor command. If the model
or its sensor inputs are unavailable, it requests the safe fixed fallback
throttle instead.

Run the host-side checks with:

```powershell
python -m unittest discover -s tests -v
```