"""Classify track turn and grade from the vehicle IMU."""

import math


TURN_UNKNOWN = "unknown"
GRADE_UNKNOWN = "unknown"
TURN_TYPES = ("straight", "left", "right", TURN_UNKNOWN)
GRADE_TYPES = ("level", "uphill", "downhill", GRADE_UNKNOWN)

TURN_ENTER_DPS = 18.0
TURN_EXIT_DPS = 8.0
GRADE_ENTER_DEG = 12.0
GRADE_EXIT_DEG = 6.0
CONFIRM_SAMPLES = 3
PITCH_GYRO_WEIGHT = 0.90
YAW_RATE_FILTER_WEIGHT = 0.25


class SectionClassifier:
    """Track stable turn and grade labels using gyro and gravity direction."""

    def __init__(self):
        self.turn_type = "straight"
        self.grade_type = "level"
        self._pitch_baseline_rad = None
        self._pitch_rad = 0.0
        self._yaw_rate_dps = 0.0
        self._last_turn_candidate = None
        self._turn_candidate_count = 0
        self._last_grade_candidate = None
        self._grade_candidate_count = 0

    def calibrate(self, pitch_baseline_rad):
        """Set the stationary, level-track pitch reference."""
        self._pitch_baseline_rad = pitch_baseline_rad
        self._pitch_rad = 0.0

    @staticmethod
    def _confirm(current, candidate, previous_candidate, count):
        if candidate == current:
            return current, None, 0
        if candidate == previous_candidate:
            count += 1
        else:
            previous_candidate = candidate
            count = 1
        if count >= CONFIRM_SAMPLES:
            return candidate, None, 0
        return current, previous_candidate, count

    def update(self, imu, elapsed_s):
        """Return turn and grade labels, or unknown labels for invalid IMU data."""
        acceleration = imu.get("acceleration_m_s2")
        gyro = imu.get("gyro_dps")
        if (
            not imu.get("imu_valid", False)
            or acceleration is None
            or gyro is None
            or self._pitch_baseline_rad is None
        ):
            return {"turn_type": TURN_UNKNOWN, "grade_type": GRADE_UNKNOWN}

        dt = max(0.001, min(float(elapsed_s), 0.1))
        yaw_rate = float(gyro[2])
        self._yaw_rate_dps = (
            (1.0 - YAW_RATE_FILTER_WEIGHT) * self._yaw_rate_dps
            + YAW_RATE_FILTER_WEIGHT * yaw_rate
        )

        if self._yaw_rate_dps >= TURN_ENTER_DPS:
            turn_candidate = "left"
        elif self._yaw_rate_dps <= -TURN_ENTER_DPS:
            turn_candidate = "right"
        elif self.turn_type == "left" and self._yaw_rate_dps >= TURN_EXIT_DPS:
            turn_candidate = "left"
        elif self.turn_type == "right" and self._yaw_rate_dps <= -TURN_EXIT_DPS:
            turn_candidate = "right"
        else:
            turn_candidate = "straight"
        (
            self.turn_type,
            self._last_turn_candidate,
            self._turn_candidate_count,
        ) = self._confirm(
            self.turn_type,
            turn_candidate,
            self._last_turn_candidate,
            self._turn_candidate_count,
        )

        accel_pitch = math.atan2(
            float(acceleration[0]),
            math.sqrt(float(acceleration[1]) ** 2 + float(acceleration[2]) ** 2),
        ) - self._pitch_baseline_rad
        self._pitch_rad += -math.radians(float(gyro[1])) * dt
        self._pitch_rad = (
            PITCH_GYRO_WEIGHT * self._pitch_rad
            + (1.0 - PITCH_GYRO_WEIGHT) * accel_pitch
        )
        pitch_deg = math.degrees(self._pitch_rad)

        if pitch_deg >= GRADE_ENTER_DEG:
            grade_candidate = "uphill"
        elif pitch_deg <= -GRADE_ENTER_DEG:
            grade_candidate = "downhill"
        elif self.grade_type == "uphill" and pitch_deg >= GRADE_EXIT_DEG:
            grade_candidate = "uphill"
        elif self.grade_type == "downhill" and pitch_deg <= -GRADE_EXIT_DEG:
            grade_candidate = "downhill"
        else:
            grade_candidate = "level"
        (
            self.grade_type,
            self._last_grade_candidate,
            self._grade_candidate_count,
        ) = self._confirm(
            self.grade_type,
            grade_candidate,
            self._last_grade_candidate,
            self._grade_candidate_count,
        )

        return {"turn_type": self.turn_type, "grade_type": self.grade_type}
