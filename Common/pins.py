"""CircuitPython pin assignments for the MSR ESP32 car."""

import board


# Set for the Wokwi ESP32 DevKitC; the physical Feather V2 uses the alternate pins.
SIMULATION = False

TRACK_V = board.IO34
ENCODER = board.IO32
I2C_SDA = board.IO22
I2C_SCL = board.IO26 if SIMULATION else board.IO20
SD_MISO = board.IO21
MOTOR_NTC = board.IO39
BATT_V = board.IO35
MOTOR_IN1 = board.IO27
MOTOR_IN2 = board.IO33
SD_SCK = board.IO5
SD_MOSI = board.IO19
SD_CS = board.IO4
LED = board.IO13
