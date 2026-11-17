"""MicroPython wrapper for the VL53L4CD time-of-flight sensor."""


OBJECT_THRESHOLD_CM = 30.0
OBJECT_CLEAR_THRESHOLD_CM = 35.0


class TOFSensor:
    """Read a measured distance from a VL53L4CD on I2C."""

    def __init__(self, i2c, address=0x29):
        self._i2c = i2c
        self._address = address
        self._overhead_detected = False

    def read(self):
        """Return a fresh distance in cm and range status, or None if unavailable."""
        try:
            raw = self._i2c.readfrom_mem(self._address, 0x0090, 2)
            distance_cm = int.from_bytes(raw, "little") / 10.0
            return {"distance_cm": distance_cm, "range_status": 0}
        except (AttributeError, OSError, ValueError):
            return None

    def new_overhead(self, distance_cm, range_status):
        """Return True once when a valid object enters the threshold."""
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
        """Release the sensor handle."""
        self._i2c = None
