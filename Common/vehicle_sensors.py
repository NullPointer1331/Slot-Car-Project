"""Analog vehicle sensors and wheel-encoder pulse counting."""

import math

import analogio
import countio


ADC_REFERENCE_V = 3.3
ADC_MAX = 65535
TRACK_DIVIDER_RATIO = 5.7
BATTERY_DIVIDER_RATIO = 2.0
NTC_BETA = 3950.0
NTC_NOMINAL_K = 298.15


class VehicleSensors:
    """Read track voltage, motor temperature, battery voltage, and wheel pulses."""

    def __init__(self, track_pin, ntc_pin, battery_pin, encoder_pin):
        self._track = analogio.AnalogIn(track_pin)
        self._ntc = analogio.AnalogIn(ntc_pin)
        self._battery = analogio.AnalogIn(battery_pin)
        self._encoder = countio.Counter(encoder_pin, edge=countio.Edge.RISE)
        self._last_pulse_count = self._encoder.count

    @staticmethod
    def _voltage(adc, divider_ratio):
        return adc.value * ADC_REFERENCE_V / ADC_MAX * divider_ratio

    def _motor_temperature_c(self):
        fraction = max(
            1 / ADC_MAX,
            min((ADC_MAX - 1) / ADC_MAX, self._ntc.value / ADC_MAX),
        )
        resistance_ratio = fraction / (1.0 - fraction)
        inverse_temperature = (
            math.log(resistance_ratio) / NTC_BETA + 1.0 / NTC_NOMINAL_K
        )
        return 1.0 / inverse_temperature - 273.15

    def read(self):
        """Return one sample; wheel_pulses is the count since the previous read."""
        pulse_count = self._encoder.count
        pulse_delta = pulse_count - self._last_pulse_count
        if pulse_delta < 0:
            pulse_delta += 2**32
        self._last_pulse_count = pulse_count
        return {
            "track_v": self._voltage(self._track, TRACK_DIVIDER_RATIO),
            "motor_temp_c": self._motor_temperature_c(),
            "battery_v": self._voltage(self._battery, BATTERY_DIVIDER_RATIO),
            "wheel_pulses": pulse_delta,
        }

    def deinit(self):
        """Release all analog inputs and the pulse counter."""
        self._track.deinit()
        self._ntc.deinit()
        self._battery.deinit()
        self._encoder.deinit()
