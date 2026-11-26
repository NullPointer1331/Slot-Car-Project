"""MicroPython wrapper for a MAX17048 LiPo fuel gauge."""


class FuelGauge:
    """Read battery voltage and state of charge from a MAX17048 on I2C."""

    def __init__(self, i2c, address=0x36):
        self._i2c = i2c
        self._address = address

    def _read_u16(self, register):
        try:
            return int.from_bytes(self._i2c.readfrom_mem(self._address, register, 2), "little")
        except (AttributeError, OSError, ValueError):
            return 0

    def read(self):
        """Return battery voltage and state of charge as a percentage."""
        voltage_raw = self._read_u16(0x02)
        percent_raw = self._read_u16(0x04)
        return {
            "voltage_v": voltage_raw * 78.125e-6,
            "percent": percent_raw / 256.0,
        }
