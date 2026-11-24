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
        self.accel_pitch_baseline_rad = None

    def _read_words(self, register, count, address=None):
        target = self._address if address is None else address
        try:
            payload = self._i2c.readfrom_mem(target, register, count * 2)
            return [
                int.from_bytes(payload[index : index + 2], "big", signed=True)
                for index in range(0, len(payload), 2)
            ]
        except (AttributeError, OSError, ValueError):
            return None

    def calibrate_gyro_bias(self, samples=100, interval_s=0.02):
        """Average stationary yaw readings and return the Z-axis bias in rad/s."""
        if samples <= 0 or interval_s < 0:
            raise ValueError(
                "Calibration samples must be positive and interval nonnegative"
            )

        bias_sum = 0.0
        pitch_sum = 0.0
        pitch_samples = 0
        for _ in range(samples):
            sample = self.read()
            gyro = sample["gyro_raw_rad_s"]
            if gyro is None:
                raise RuntimeError("ICM-20948 gyro data unavailable during calibration")
            bias_sum += gyro[2]
            acceleration = sample["acceleration_m_s2"]
            if acceleration is not None:
                pitch_sum += math.atan2(
                    acceleration[0],
                    math.sqrt(acceleration[1] ** 2 + acceleration[2] ** 2),
                )
                pitch_samples += 1
            time.sleep(interval_s)
        if pitch_samples == 0:
            raise RuntimeError(
                "ICM-20948 accelerometer data unavailable during calibration"
            )
        self._gyro_z_bias_rad_s = bias_sum / samples
        self.accel_pitch_baseline_rad = pitch_sum / pitch_samples
        return self._gyro_z_bias_rad_s

    def read(self):
        """Return motion and environment readings in documented units."""
        accel_raw = self._read_words(0x2D, 3)
        gyro_raw = self._read_words(0x33, 3)
        mag_raw = self._read_words(0x10, 3, self._mag_address)

        acceleration = (
            tuple(value / 16384.0 * 9.80665 for value in accel_raw)
            if accel_raw is not None
            else None
        )
        raw_gyro_rad_s = (
            tuple(value / 131.0 * math.pi / 180.0 for value in gyro_raw)
            if gyro_raw is not None
            else None
        )
        gyro_rad_s = (
            (raw_gyro_rad_s[0], raw_gyro_rad_s[1], raw_gyro_rad_s[2] - self._gyro_z_bias_rad_s)
            if raw_gyro_rad_s is not None
            else None
        )
        gyro_dps = tuple(math.degrees(value) for value in gyro_rad_s) if gyro_rad_s is not None else None
        acceleration_g = (
            tuple(value / 9.80665 for value in acceleration)
            if acceleration is not None
            else None
        )
        magnetic_field = (
            tuple(value / 10.0 for value in mag_raw)
            if mag_raw is not None
            else None
        )

        return {
            "acceleration_m_s2": acceleration,
            "acceleration_g": acceleration_g,
            "gyro_raw_rad_s": raw_gyro_rad_s,
            "gyro_rad_s": gyro_rad_s,
            "gyro_dps": gyro_dps,
            "magnetic_field_uT": magnetic_field,
            "imu_valid": acceleration is not None and gyro_dps is not None,
        }
