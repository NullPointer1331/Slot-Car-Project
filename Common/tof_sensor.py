"""Input/output wrapper for the Adafruit VL53L4CD time-of-flight sensor."""

import adafruit_vl53l4cd


OBJECT_THRESHOLD_CM = 30.0
OBJECT_CLEAR_THRESHOLD_CM = 35.0
class TOFSensor:
    """Read non-blocking distance measurements from a VL53L4CD on I2C."""

    def __init__(self, i2c, address=0x29):
        self._sensor = adafruit_vl53l4cd.VL53L4CD(i2c, address=address)
        self._sensor.start_ranging()
        self._overhead_detected = False

    def read(self):
        """Return a fresh distance in cm and range status, or None if not ready."""
        if not self._sensor.data_ready:
            return None

        self._sensor.clear_interrupt()
        return {
            "distance_cm": self._sensor.distance,
            "range_status": self._sensor.range_status,
        }

    def new_overhead(
        self, distance_cm, range_status):
        """Return True once when a valid object enters the threshold.

        A reading at or beyond ``clear_threshold_cm`` rearms detection. Invalid
        range readings also rearm it, so each new object can be counted once.
        """
        if OBJECT_THRESHOLD_CM <= 0 or OBJECT_CLEAR_THRESHOLD_CM <= OBJECT_THRESHOLD_CM:
            raise ValueError(
                "Clear threshold must be greater than a positive detection threshold"
            )

        if range_status != 0:
            self._overhead_detected = False
            return False

        if self._overhead_detected:
            if distance_cm >= OBJECT_CLEAR_THRESHOLD_CM:
                self._overhead_detected = False
            return False

        if distance_cm <= OBJECT_THRESHOLD_CM:
            self._overhead_detected = True
            return True

        return False

    def deinit(self):
        """Stop ranging and release the sensor."""
        self._sensor.stop_ranging()
