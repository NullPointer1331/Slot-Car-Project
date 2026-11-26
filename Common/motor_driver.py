"""MicroPython driver wrapper for a DRV8871 DC motor driver."""

try:
    import machine
except ImportError:  # pragma: no cover - host builds do not use this module
    machine = None


class MotorDriver:
    """Drive a brushed DC motor using a pair of PWM-controlled inputs."""

    def __init__(self, in1_pin, in2_pin, frequency=20000):
        if machine is None:
            raise RuntimeError("MicroPython's machine module is required")
        self._in1 = machine.PWM(machine.Pin(in1_pin))
        self._in2 = machine.PWM(machine.Pin(in2_pin))
        self._in1.freq(frequency)
        self._in2.freq(frequency)
        self.stop()

    @staticmethod
    def _set_duty(pwm, duty):
        duty = max(0.0, min(1.0, duty))
        scaled = int(round(duty * 65535))
        if hasattr(pwm, "duty_u16"):
            pwm.duty_u16(scaled)
        elif hasattr(pwm, "duty"):
            pwm.duty(scaled)
        else:  # pragma: no cover - defensive fallback for minimal ports
            pwm.duty(scaled)

    def set_speed(self, speed):
        """Apply a signed speed from -1.0 (reverse) to 1.0 (forward)."""
        speed = max(-1.0, min(1.0, speed))
        if speed > 0:
            self._set_duty(self._in1, abs(speed))
            self._set_duty(self._in2, 0.0)
        elif speed < 0:
            self._set_duty(self._in1, 0.0)
            self._set_duty(self._in2, abs(speed))
        else:
            self.stop()
        return speed

    def stop(self):
        """Set both driver inputs low to coast."""
        self._set_duty(self._in1, 0.0)
        self._set_duty(self._in2, 0.0)

    def brake(self):
        """Apply the DRV8871 brake state (IN1 low, IN2 high)."""
        self._set_duty(self._in1, 0.0)
        self._set_duty(self._in2, 1.0)

    def deinit(self):
        """Brake the motor and release both PWM outputs."""
        self.brake()
        self._in1.deinit()
        self._in2.deinit()
