"""Input/output wrapper for the Adafruit ICM-20948 IMU."""

import math
import time

import adafruit_icm20x


class IMU:
    """Read motion data from an ICM-20948 on an existing I2C bus."""

    def __init__(self, i2c, address=0x69):
        self._sensor = adafruit_icm20x.ICM20948(i2c, address=address)
        self._gyro_z_bias_rad_s = 0.0

    def calibrate_gyro_bias(self, samples=100, interval_s=0.02):
        """Average stationary yaw readings and return the Z-axis bias in rad/s."""
        if samples <= 0 or interval_s < 0:
            raise ValueError(
                "Calibration samples must be positive and interval nonnegative"
            )

        bias_sum = 0.0
        for _ in range(samples):
            gyro = self._sensor.gyro
            if gyro is None:
                raise RuntimeError("ICM-20948 gyro data unavailable during calibration")
            bias_sum += gyro[2]
            time.sleep(interval_s)
        self._gyro_z_bias_rad_s = bias_sum / samples
        return self._gyro_z_bias_rad_s

    def read(self):
        """Return motion and environment readings in documented units."""
        acceleration = self._sensor.acceleration
        raw_gyro_rad_s = self._sensor.gyro
        gyro_rad_s = (
            (
                raw_gyro_rad_s[0],
                raw_gyro_rad_s[1],
                raw_gyro_rad_s[2] - self._gyro_z_bias_rad_s,
            )
            if raw_gyro_rad_s is not None
            else None
        )
        gyro_dps = (
            tuple(math.degrees(value) for value in gyro_rad_s)
            if gyro_rad_s is not None
            else None
        )
        return {
            "acceleration_m_s2": acceleration,
            "acceleration_g": (
                tuple(value / 9.80665 for value in acceleration)
                if acceleration is not None
                else None
            ),
            "gyro_raw_rad_s": raw_gyro_rad_s,
            "gyro_rad_s": gyro_rad_s,
            "gyro_dps": gyro_dps,
            "magnetic_field_uT": self._sensor.magnetic,
            "temperature_c": self._sensor.temperature,
        }
