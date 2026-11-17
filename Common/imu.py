"""MicroPython ICM-20948 IMU wrapper."""

import math
import time


class IMU:
    """Read motion data from an ICM-20948 on an existing I2C bus."""

    def __init__(self, i2c, address=0x69):
        self._i2c = i2c
        self._address = address
        self._mag_address = 0x0C
        self._gyro_z_bias_rad_s = 0.0

    def _read_words(self, register, count, address=None):
        target = self._address if address is None else address
        try:
            payload = self._i2c.readfrom_mem(target, register, count * 2)
            return [
                int.from_bytes(payload[index : index + 2], "big", signed=True)
                for index in range(0, len(payload), 2)
            ]
        except (AttributeError, OSError, ValueError):
            return [0] * count

    def calibrate_gyro_bias(self, samples=100, interval_s=0.02):
        """Average stationary yaw readings and return the Z-axis bias in rad/s."""
        if samples <= 0 or interval_s < 0:
            raise ValueError(
                "Calibration samples must be positive and interval nonnegative"
            )

        bias_sum = 0.0
        for _ in range(samples):
            gyro = self.read()["gyro_raw_rad_s"]
            if gyro is None:
                raise RuntimeError("ICM-20948 gyro data unavailable during calibration")
            bias_sum += gyro[2]
            time.sleep(interval_s)
        self._gyro_z_bias_rad_s = bias_sum / samples
        return self._gyro_z_bias_rad_s

    def read(self):
        """Return motion and environment readings in documented units."""
        try:
            accel_raw = self._read_words(0x2D, 3)
            gyro_raw = self._read_words(0x33, 3)
            mag_raw = self._read_words(0x10, 3, self._mag_address)
        except Exception:
            accel_raw = [0, 0, 0]
            gyro_raw = [0, 0, 0]
            mag_raw = [0, 0, 0]

        acceleration = tuple(value / 16384.0 * 9.80665 for value in accel_raw)
        raw_gyro_rad_s = tuple(value / 131.0 * math.pi / 180.0 for value in gyro_raw)
        gyro_rad_s = (
            (raw_gyro_rad_s[0], raw_gyro_rad_s[1], raw_gyro_rad_s[2] - self._gyro_z_bias_rad_s)
            if raw_gyro_rad_s is not None
            else None
        )
        gyro_dps = tuple(math.degrees(value) for value in gyro_rad_s) if gyro_rad_s is not None else None

        return {
            "acceleration_m_s2": acceleration,
            "acceleration_g": tuple(value / 9.80665 for value in acceleration),
            "gyro_raw_rad_s": raw_gyro_rad_s,
            "gyro_rad_s": gyro_rad_s,
            "gyro_dps": gyro_dps,
            "magnetic_field_uT": tuple(value / 10.0 for value in mag_raw),
        }
