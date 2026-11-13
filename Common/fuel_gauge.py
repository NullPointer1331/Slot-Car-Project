"""Input/output wrapper for the Adafruit MAX17048 LiPo fuel gauge."""

import adafruit_max1704x


class FuelGauge:
    """Read battery voltage and state of charge from a MAX17048 on I2C."""

    def __init__(self, i2c, address=0x36):
        self._sensor = adafruit_max1704x.MAX17048(i2c, address=address)

    def read(self):
        """Return battery voltage and state of charge as a percentage."""
        return {
            "voltage_v": self._sensor.cell_voltage,
            "percent": self._sensor.cell_percent,
        }
