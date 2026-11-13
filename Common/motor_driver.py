"""Input/output wrapper for the Adafruit DRV8871 DC motor driver."""

import pwmio

class MotorDriver:
    """Drive one brushed DC motor using the driver's IN1 and IN2 PWM inputs."""

    def __init__(self, in1_pin, in2_pin, frequency=20_000):
        self._in1 = pwmio.PWMOut(in1_pin, frequency=frequency, duty_cycle=0)
        self._in2 = pwmio.PWMOut(in2_pin, frequency=frequency, duty_cycle=0)

    def set_speed(self, speed):
        """Apply signed speed from -1.0 (reverse) to 1.0 (forward).

        Returns the clamped speed command sent to the driver.
        """
        speed = max(-1.0, min(1.0, speed))
        duty = int(abs(speed) * 65535)

        if speed > 0:
            self._in1.duty_cycle = duty
            self._in2.duty_cycle = 0
        elif speed < 0:
            self._in1.duty_cycle = 0
            self._in2.duty_cycle = duty
        else:
            self.stop()

        return speed

    def stop(self):
        """Set both driver inputs low to coast."""
        self._in1.duty_cycle = 0
        self._in2.duty_cycle = 0

    def brake(self):
        """Apply the DRV8871 brake state (IN1 low, IN2 high)."""
        self._in1.duty_cycle = 0
        self._in2.duty_cycle = 65535

    def deinit(self):
        """Brake the motor and release both PWM outputs."""
        self.brake()
        self._in1.deinit()
        self._in2.deinit()
