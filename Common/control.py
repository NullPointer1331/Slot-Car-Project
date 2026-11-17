"""Shared vehicle safety limits."""

TRACK_OK_V = 9.0
CRASH_HOLD_MS = 500
CRASH_G = 1.5
TEMP_DERATE_C = 80.0
TEMP_CUTOFF_C = 100.0
BATT_PIT_V = 3.5
BATT_STOP_V = 3.3
DEFAULT_THROTTLE_CAP = 0.90
RECOVERY_THROTTLE = 0.30
MAX_THROTTLE_RISE_PER_LOOP = 0.05

BOOT = "BOOT"
CALIBRATE = "CALIBRATE"
WAIT = "WAIT"
RACE = "RACE"
CRASH = "CRASH"
RECOVER = "RECOVER"


class VehicleController:
    """Read vehicle sensors, enforce safety limits, and control the motor."""

    def __init__(self, monitor, motor):
        self.monitor = monitor
        self.motor = motor
        self.state = BOOT
        self.throttle_cap = DEFAULT_THROTTLE_CAP
        self._lost_since_ms = None
        self.throttle_command = 0.0
        self.motor.brake()

    def calibrate(self):
        """Calibrate sensors and then wait for track power."""
        if self.state != BOOT:
            raise RuntimeError("Calibration can only begin from BOOT")
        self.state = CALIBRATE
        self.monitor.calibrate()
        self.state = WAIT

    def read(self):
        """Read a unified sensor sample from the configured monitor."""
        return self.monitor.read()

    def _safety_cap(self, sample):
        cap = max(0.0, min(1.0, self.throttle_cap))
        if sample["track_v"] < TRACK_OK_V:
            return 0.0
        if sample["motor_temp_c"] > TEMP_DERATE_C:
            cap = min(cap, 0.5)
        if (
            sample["motor_temp_c"] > TEMP_CUTOFF_C
            or sample["battery_v"] < BATT_STOP_V
        ):
            return 0.0
        return cap

    def update(self, now_ms, sample, requested_throttle):
        """Check a throttle request against current sensors and apply it."""
        events = []
        track_v = sample["track_v"]
        requested_throttle = max(0.0, min(1.0, requested_throttle))

        if self.state == CRASH and track_v >= TRACK_OK_V:
            self._lost_since_ms = None
            self.state = RECOVER
            events.append("STATE -> RECOVER")
        elif self.state == WAIT and track_v > TRACK_OK_V:
            self.state = RACE
            events.append("STATE -> RACE (track power detected)")

        if self.state in (RACE, RECOVER):
            if track_v < TRACK_OK_V:
                if self._lost_since_ms is None:
                    self._lost_since_ms = now_ms
                elif now_ms - self._lost_since_ms > CRASH_HOLD_MS:
                    self.state = CRASH
                    self.throttle_cap = max(0.2, self.throttle_cap - 0.05)
                    events.append("STATE -> CRASH, throttle cap reduced")
            elif self._lost_since_ms is not None:
                events.append(
                    "Short dip {:.0f} ms, still racing".format(
                        now_ms - self._lost_since_ms
                    )
                )
                self._lost_since_ms = None

        acceleration_g = sample["acceleration_g"]
        if (
            self.state == RACE
            and acceleration_g is not None
            and abs(acceleration_g[1]) > CRASH_G
        ):
            self.state = CRASH
            events.append("STATE -> CRASH (lateral g spike)")
        elif self.state == RECOVER and sample["new_landmark"]:
            self.state = RACE
            events.append("STATE -> RACE (position reset)")

        cap = self._safety_cap(sample)
        if self.state == RACE:
            target_throttle = min(requested_throttle, cap)
        elif self.state == RECOVER:
            target_throttle = min(RECOVERY_THROTTLE, cap)
        else:
            target_throttle = 0.0

        if target_throttle > self.throttle_command:
            self.throttle_command = min(
                target_throttle,
                self.throttle_command + MAX_THROTTLE_RISE_PER_LOOP,
            )
        else:
            self.throttle_command = target_throttle

        if self.throttle_command == 0:
            self.motor.brake()
        else:
            self.motor.set_speed(self.throttle_command)

        return {
            "state": self.state,
            "safety_cap": cap,
            "target_throttle": target_throttle,
            "throttle_command": self.throttle_command,
            "events": events,
        }

    def deinit(self):
        """Stop the motor and release the monitor and motor resources."""
        try:
            self.motor.brake()
        finally:
            try:
                self.monitor.deinit()
            finally:
                self.motor.deinit()
