"""Drive the car and log IMU and time-of-flight sensor readings."""

import time

import board

from Common.imu import IMU
from Common.motor_driver import MotorDriver
from Common.sd_logger import SDCardCSVLogger
from Common.tof_sensor import TOFSensor


# Change these to PWM-capable GPIO pins exposed by your ESP32 board.
MOTOR_IN1_PIN = board.IO25
MOTOR_IN2_PIN = board.IO26
# SD breakout CS; SCK, MOSI, and MISO use the Feather's default SPI pins.
SD_CHIP_SELECT_PIN = board.D14

INITIAL_SPEED = 0.10
SPEED_INCREMENT = 0.05
DETECTIONS_PER_INCREMENT = 2
MAX_SPEED = 1.0
LOG_FILE = "/sd/data.csv"
LOOP_DELAY_S = 0.05
LOG_FLUSH_INTERVAL_S = 1.0

CSV_HEADER = (
    "elapsed_s,heading_deg,roll_deg,pitch_deg,"
    "quaternion_w,quaternion_x,quaternion_y,quaternion_z,"
    "accel_x_m_s2,accel_y_m_s2,accel_z_m_s2,"
    "linear_accel_x_m_s2,linear_accel_y_m_s2,linear_accel_z_m_s2,"
    "gravity_x_m_s2,gravity_y_m_s2,gravity_z_m_s2,"
    "gyro_x_dps,gyro_y_dps,gyro_z_dps,"
    "magnetic_x_uT,magnetic_y_uT,magnetic_z_uT,temperature_c,"
    "distance_cm,range_status,new_object,object_count,speed_command\n"
)


def vector_fields(vector, length):
    """Return vector components, or empty CSV cells when a reading is unavailable."""
    if vector is None:
        return [""] * length
    return list(vector)


i2c = board.I2C()
motor = MotorDriver(MOTOR_IN1_PIN, MOTOR_IN2_PIN)
logger = None

try:
    imu = IMU(i2c)
    distance_sensor = TOFSensor(i2c)
    logger = SDCardCSVLogger(
        board.SPI(), SD_CHIP_SELECT_PIN, LOG_FILE, CSV_HEADER
    )

    start_time = time.monotonic()
    last_flush = start_time
    current_speed = INITIAL_SPEED
    object_count = 0
    motor.set_speed(current_speed)

    while True:
        imu_data = imu.read()
        distance_data = distance_sensor.read()
        new_object = False

        if distance_data is not None:
            new_object = distance_sensor.new_overhead(
                distance_data["distance_cm"],
                distance_data["range_status"]
            )
            if new_object:
                object_count += 1
                if object_count % DETECTIONS_PER_INCREMENT == 0:
                    current_speed = min(current_speed + SPEED_INCREMENT, MAX_SPEED)
                    motor.set_speed(current_speed)

        orientation = imu_data["euler_deg"]
        row = [
            "{:.3f}".format(time.monotonic() - start_time),
            *vector_fields(orientation, 3),
            *vector_fields(imu_data["quaternion"], 4),
            *vector_fields(imu_data["acceleration_m_s2"], 3),
            *vector_fields(imu_data["linear_acceleration_m_s2"], 3),
            *vector_fields(imu_data["gravity_m_s2"], 3),
            *vector_fields(imu_data["gyro_dps"], 3),
            *vector_fields(imu_data["magnetic_field_uT"], 3),
            imu_data["temperature_c"],
            distance_data["distance_cm"] if distance_data is not None else "",
            distance_data["range_status"] if distance_data is not None else "",
            1 if new_object else 0,
            object_count,
            "{:.2f}".format(current_speed),
        ]
        logger.write_row(row)

        now = time.monotonic()
        if now - last_flush >= LOG_FLUSH_INTERVAL_S:
            logger.flush()
            last_flush = now

        if orientation is None or any(value is None for value in orientation):
            print("Waiting for BNO055 orientation data...")
        else:
            heading, roll, pitch = orientation
            print(
                "heading={:.1f} roll={:.1f} pitch={:.1f} "
                "objects={} speed={:.2f}".format(
                    heading, roll, pitch, object_count, current_speed
                )
            )

        time.sleep(LOOP_DELAY_S)
finally:
    motor.deinit()
    i2c.deinit()
    if logger is not None:
        logger.deinit()
