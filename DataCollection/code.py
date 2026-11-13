"""Capture labelled sensor and throttle samples for car-control model training."""

import time

import busio
import digitalio

from Common.control import (
    BATT_PIT_V,
    CRASH,
    TEMP_DERATE_C,
    VehicleController,
)
from Common.monitor import Monitor
from Common.motor_driver import MotorDriver
from Common.pins import (
    BATT_V,
    ENCODER,
    I2C_SCL,
    I2C_SDA,
    LED,
    MOTOR_IN1,
    MOTOR_IN2,
    MOTOR_NTC,
    SD_CS,
    SD_MISO,
    SD_MOSI,
    SD_SCK,
    TRACK_V,
)
from Common.sd_logger import SDCardCSVLogger


LOG_FILE = "/sd/data.csv"
LOOP_PERIOD_S = 0.02
LOG_FLUSH_INTERVAL_S = 1.0
INITIAL_THROTTLE = 0.10
THROTTLE_INCREMENT = 0.05
LANDMARKS_PER_INCREMENT = 2
MAX_THROTTLE = 1.0

CSV_HEADER = (
    "elapsed_s,landmark_count,distance_m,wheel_pulses,speed_m_s,"
    "accel_x_m_s2,accel_y_m_s2,accel_z_m_s2,"
    "accel_x_g,accel_y_g,accel_z_g,"
    "gyro_x_dps,gyro_y_dps,gyro_z_dps,"
    "magnetic_x_uT,magnetic_y_uT,magnetic_z_uT,imu_temp_c,"
    "track_v,motor_temp_c,battery_v,fuel_gauge_voltage_v,fuel_percent,"
    "distance_cm,range_status,new_landmark,"
    "state,requested_throttle,throttle_command,safety_cap\n"
)


def vector_fields(vector, length):
    """Return vector components, or empty CSV cells when unavailable."""
    if vector is None:
        return [""] * length
    return list(vector)


def log_sample(
    logger,
    sample,
    control_data,
    requested_throttle,
    elapsed_s,
    last_flush,
    force_flush=False,
):
    """Write one training sample and flush periodically."""
    row = [
        "{:.3f}".format(elapsed_s),
        sample["landmark_count"],
        "{:.3f}".format(sample["distance_m"]),
        sample["wheel_pulses"],
        "{:.3f}".format(sample["speed_m_s"]),
        *vector_fields(sample["acceleration_m_s2"], 3),
        *vector_fields(sample["acceleration_g"], 3),
        *vector_fields(sample["gyro_dps"], 3),
        *vector_fields(sample["magnetic_field_uT"], 3),
        sample["temperature_c"],
        "{:.3f}".format(sample["track_v"]),
        "{:.2f}".format(sample["motor_temp_c"]),
        "{:.3f}".format(sample["battery_v"]),
        "{:.3f}".format(sample["fuel_gauge_voltage_v"]),
        "{:.1f}".format(sample["fuel_percent"]),
        sample["distance_cm"] if sample["distance_cm"] is not None else "",
        sample["range_status"] if sample["range_status"] is not None else "",
        1 if sample["new_landmark"] else 0,
        control_data["state"],
        "{:.2f}".format(requested_throttle),
        "{:.2f}".format(control_data["throttle_command"]),
        "{:.2f}".format(control_data["safety_cap"]),
    ]
    logger.write_row(row)
    now = time.monotonic()
    if force_flush or now - last_flush >= LOG_FLUSH_INTERVAL_S:
        logger.flush()
        return now
    return last_flush


def deinitialize(controller, motor, monitor, status_led, i2c, logger, spi):
    """Release hardware and storage resources, attempting every cleanup."""
    try:
        if controller is not None:
            controller.deinit()
        else:
            try:
                if monitor is not None:
                    monitor.deinit()
            finally:
                if motor is not None:
                    try:
                        motor.brake()
                    finally:
                        motor.deinit()
    finally:
        try:
            if status_led is not None:
                status_led.value = False
                status_led.deinit()
        finally:
            try:
                i2c.deinit()
            finally:
                try:
                    if logger is not None:
                        logger.deinit()
                finally:
                    if spi is not None:
                        spi.deinit()


def main():
    i2c = busio.I2C(I2C_SCL, I2C_SDA, frequency=400000)
    motor = None
    monitor = None
    controller = None
    status_led = None
    logger = None
    spi = None

    try:
        motor = MotorDriver(MOTOR_IN1, MOTOR_IN2)
        status_led = digitalio.DigitalInOut(LED)
        status_led.switch_to_output(value=False)
        monitor = Monitor(i2c, TRACK_V, MOTOR_NTC, BATT_V, ENCODER)

        controller = VehicleController(monitor, motor)
        controller.calibrate()

        spi = busio.SPI(SD_SCK, MOSI=SD_MOSI, MISO=SD_MISO)
        logger = SDCardCSVLogger(spi, SD_CS, LOG_FILE, CSV_HEADER)
        start_time = time.monotonic()
        last_flush = start_time
        last_status = start_time
        pit_announced = False
        fault_active = False

        while True:
            sample_time = time.monotonic()
            sample = controller.read()
            landmark_count = sample["landmark_count"]
            requested_throttle = min(
                INITIAL_THROTTLE
                + (landmark_count // LANDMARKS_PER_INCREMENT)
                * THROTTLE_INCREMENT,
                MAX_THROTTLE,
            )
            control_data = controller.update(
                now_ms=(sample_time - start_time) * 1000,
                sample=sample,
                requested_throttle=requested_throttle,
            )
            state = control_data["state"]
            cap = control_data["safety_cap"]
            throttle_command = control_data["throttle_command"]

            for event in control_data["events"]:
                print(event)

            fault = (
                state == CRASH
                or sample["motor_temp_c"] > TEMP_DERATE_C
                or sample["battery_v"] < BATT_PIT_V
            )
            new_fault = fault and not fault_active
            fault_active = fault
            blink_period_s = 0.1 if fault else 0.5
            status_led.value = int(sample_time / blink_period_s) % 2 == 0

            last_flush = log_sample(
                logger,
                sample,
                control_data,
                requested_throttle,
                sample_time - start_time,
                last_flush,
                force_flush=new_fault,
            )

            battery_low = sample["battery_v"] < BATT_PIT_V
            if battery_low and not pit_announced:
                print("PIT: LiPo {:.2f} V".format(sample["battery_v"]))
            pit_announced = battery_low
            if sample_time - last_status >= 5.0:
                print(
                    "landmarks={} distance={:.2f} m speed={:.2f} m/s "
                    "state={} throttle={:.2f} cap={:.2f}".format(
                        landmark_count,
                        sample["distance_m"],
                        sample["speed_m_s"],
                        state,
                        throttle_command,
                        cap,
                    )
                )
                last_status = sample_time

            time.sleep(max(0, LOOP_PERIOD_S - (time.monotonic() - sample_time)))
    finally:
        deinitialize(controller, motor, monitor, status_led, i2c, logger, spi)


main()
