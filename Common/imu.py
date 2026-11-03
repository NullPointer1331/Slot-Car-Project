"""Input/output wrapper for the Adafruit BNO055 IMU."""

import adafruit_bno055


class IMU:
    """Read orientation and motion data from a BNO055 on an existing I2C bus."""

    def __init__(self, i2c, address=0x28):
        self._sensor = adafruit_bno055.BNO055_I2C(i2c, address=address)

    def read(self):
        """Return the latest sensor outputs, with vector values in documented units."""
        return {
            "euler_deg": self._sensor.euler,
            "quaternion": self._sensor.quaternion,
            "acceleration_m_s2": self._sensor.acceleration,
            "linear_acceleration_m_s2": self._sensor.linear_acceleration,
            "gravity_m_s2": self._sensor.gravity,
            "gyro_dps": self._sensor.gyro,
            "magnetic_field_uT": self._sensor.magnetic,
            "temperature_c": self._sensor.temperature,
        }
