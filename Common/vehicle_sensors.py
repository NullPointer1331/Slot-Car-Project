"""MicroPython analog sensor and wheel-encoder wrapper."""

import math

try:
    import machine
except ImportError:  # pragma: no cover - host builds do not use this module
    machine = None


ADC_REFERENCE_V = 3.3
ADC_MAX = 65535
TRACK_DIVIDER_RATIO = 5.7
BATTERY_DIVIDER_RATIO = 2.0
NTC_BETA = 3950.0
NTC_NOMINAL_K = 298.15


class VehicleSensors:
    """Read track voltage, motor temperature, battery voltage, and wheel pulses."""

    def __init__(self, track_pin, ntc_pin, battery_pin, encoder_pin):
        if machine is None:
            raise RuntimeError("MicroPython's machine module is required")

        self._track = machine.ADC(machine.Pin(track_pin))
        self._ntc = machine.ADC(machine.Pin(ntc_pin))
        self._battery = machine.ADC(machine.Pin(battery_pin))
        self._encoder = machine.Pin(encoder_pin, machine.Pin.IN)
        self._pulse_count = 0
        self._last_pulse_count = 0

        for adc in (self._track, self._ntc, self._battery):
            try:
                adc.atten(machine.ADC.ATTN_11DB)
            except (AttributeError, TypeError, ValueError):
                pass
            try:
                adc.width(machine.ADC.WIDTH_12BIT)
            except (AttributeError, TypeError, ValueError):
                pass

        self._encoder.irq(trigger=machine.Pin.IRQ_RISING, handler=self._on_encoder)

    def _on_encoder(self, pin):
        self._pulse_count += 1

    @staticmethod
    def _read_adc(adc):
        if hasattr(adc, "read_u16"):
            return adc.read_u16()
        if hasattr(adc, "read"):
            return adc.read()
        raise RuntimeError("ADC does not expose a read method")

    @staticmethod
    def _voltage(adc, divider_ratio):
        return (VehicleSensors._read_adc(adc) / ADC_MAX) * ADC_REFERENCE_V * divider_ratio

    def _motor_temperature_c(self):
        ntc_value = self._read_adc(self._ntc)
        fraction = max(1 / ADC_MAX, min((ADC_MAX - 1) / ADC_MAX, ntc_value / ADC_MAX))
        resistance_ratio = fraction / (1.0 - fraction)
        inverse_temperature = math.log(resistance_ratio) / NTC_BETA + 1.0 / NTC_NOMINAL_K
        return 1.0 / inverse_temperature - 273.15

    def read(self):
        """Return one sample; wheel_pulses is the count since the previous read."""
        pulse_delta = self._pulse_count - self._last_pulse_count
        self._last_pulse_count = self._pulse_count
        return {
            "track_v": self._voltage(self._track, TRACK_DIVIDER_RATIO),
            "motor_temp_c": self._motor_temperature_c(),
            "battery_v": self._voltage(self._battery, BATTERY_DIVIDER_RATIO),
            "wheel_pulses": pulse_delta,
        }

    def deinit(self):
        """Release all analog inputs and the pulse counter."""
        self._encoder.irq(handler=None)
        self._track = None
        self._ntc = None
        self._battery = None
