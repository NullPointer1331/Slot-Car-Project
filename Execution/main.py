"""Run the exported throttle model at 50 Hz behind the vehicle safety layer."""

import time

try:
    import machine
except ImportError:
    machine = None

from Common.clock import monotonic
from Common.control import VehicleController
from Common.model_inference import ThrottleModel
from Common.monitor import Monitor
from Common.motor_driver import MotorDriver
from Common.pins import (
    BATT_V,
    ENCODER,
    I2C_SCL,
    I2C_SDA,
    MOTOR_IN1,
    MOTOR_IN2,
    MOTOR_NTC,
    TRACK_V,
)


LOOP_PERIOD_S = 0.02
SAFE_FALLBACK_THROTTLE = 0.30
MODEL_PATH = "/throttle_model.json"


def feature_values(model, sample):
    """Build the model feature vector in the exported feature order."""
    if not sample.get("imu_valid", False):
        raise ValueError("IMU feature sample is invalid")
    turn_type = sample["turn_type"]
    grade_type = sample["grade_type"]
    if turn_type not in ("straight", "left", "right"):
        raise ValueError("Unknown turn type: {}".format(turn_type))
    if grade_type not in ("level", "uphill", "downhill"):
        raise ValueError("Unknown grade type: {}".format(grade_type))
    acceleration = sample["acceleration_m_s2"]
    gyro = sample["gyro_dps"]
    values = {
        "turn_left": 1.0 if turn_type == "left" else 0.0,
        "turn_right": 1.0 if turn_type == "right" else 0.0,
        "grade_uphill": 1.0 if grade_type == "uphill" else 0.0,
        "grade_downhill": 1.0 if grade_type == "downhill" else 0.0,
        "lap_distance_m": sample["lap_distance_m"],
        "total_distance_m": sample["total_distance_m"],
        "speed_m_s": sample["speed_m_s"],
        "accel_x_m_s2": acceleration[0] if acceleration else 0.0,
        "accel_y_m_s2": acceleration[1] if acceleration else 0.0,
        "accel_z_m_s2": acceleration[2] if acceleration else 0.0,
        "gyro_x_dps": gyro[0] if gyro else 0.0,
        "gyro_y_dps": gyro[1] if gyro else 0.0,
        "gyro_z_dps": gyro[2] if gyro else 0.0,
        "previous_throttle": sample["previous_throttle"],
    }
    return [values[name] for name in model.features]


def main():
    if machine is None:
        raise RuntimeError("This sketch requires MicroPython's machine module")

    i2c = machine.I2C(
        0,
        scl=machine.Pin(I2C_SCL),
        sda=machine.Pin(I2C_SDA),
        freq=400000,
    )
    motor = None
    monitor = None
    controller = None

    try:
        motor = MotorDriver(MOTOR_IN1, MOTOR_IN2)
        monitor = Monitor(i2c, TRACK_V, MOTOR_NTC, BATT_V, ENCODER)
        controller = VehicleController(monitor, motor)
        controller.calibrate()

        try:
            model = ThrottleModel(MODEL_PATH)
            model_error = None
        except Exception as error:
            model = None
            model_error = error

        if model_error is not None:
            print("MODEL unavailable; using safe fixed throttle:", model_error)

        previous_throttle = 0.0
        model_input_fault_active = False
        start_time = monotonic()
        while True:
            sample_time = monotonic()
            sample = controller.read()
            sample["previous_throttle"] = previous_throttle

            requested_throttle = SAFE_FALLBACK_THROTTLE
            if model is not None:
                try:
                    requested_throttle = model.predict(feature_values(model, sample))
                except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
                    requested_throttle = SAFE_FALLBACK_THROTTLE
                    if not model_input_fault_active:
                        print("MODEL input unavailable; using safe fallback:", error)
                    model_input_fault_active = True
                else:
                    if model_input_fault_active:
                        print("MODEL inputs recovered")
                    model_input_fault_active = False

            control_data = controller.update(
                now_ms=(sample_time - start_time) * 1000,
                sample=sample,
                requested_throttle=requested_throttle,
            )
            previous_throttle = control_data["throttle_command"]

            time.sleep(max(0, LOOP_PERIOD_S - (monotonic() - sample_time)))
    finally:
        if controller is not None:
            controller.deinit()
        elif monitor is not None:
            monitor.deinit()
        elif motor is not None:
            motor.deinit()


if __name__ == "__main__":
    main()