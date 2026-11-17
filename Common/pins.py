"""MicroPython pin assignments for the MSR ESP32 car."""

# MicroPython uses the GPIO number directly instead of CircuitPython's board.IOxx
# names. Keep the same physical layout as the original CircuitPython project.
SIMULATION = False

TRACK_V = 34
ENCODER = 32
I2C_SDA = 22
I2C_SCL = 26 if SIMULATION else 20
SD_MISO = 21
MOTOR_NTC = 39
BATT_V = 35
MOTOR_IN1 = 27
MOTOR_IN2 = 33
SD_SCK = 5
SD_MOSI = 19
SD_CS = 4
LED = 13
