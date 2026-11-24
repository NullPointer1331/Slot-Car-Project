"""Unified readings and landmark tracking for the car's sensors."""

from Common.clock import monotonic
from Common.fuel_gauge import FuelGauge
from Common.imu import IMU
from Common.section_classifier import SectionClassifier
from Common.tof_sensor import TOFSensor
from Common.vehicle_sensors import VehicleSensors


MM_PER_PULSE = 15.7


class Monitor:
    """Read all vehicle sensors and track wheel distance and landmarks."""

    def __init__(
        self,
        i2c,
        track_pin,
        ntc_pin,
        battery_pin,
        encoder_pin,
        tof_address=0x29,
        fuel_gauge_address=0x36,
        imu_address=0x69,
    ):
        self._imu = IMU(i2c, address=imu_address)
        self._tof = TOFSensor(i2c, address=tof_address)
        self._fuel_gauge = FuelGauge(i2c, address=fuel_gauge_address)
        self._vehicle = VehicleSensors(
            track_pin, ntc_pin, battery_pin, encoder_pin
        )
        self.landmark_count = 0
        self.lap_id = 0
        self.lap_distance_m = 0.0
        self.total_distance_m = 0.0
        self.speed_m_s = 0.0
        self._section_classifier = SectionClassifier()
        self._last_sample_time = monotonic()

    def calibrate(self):
        """Calibrate gyro bias and level-track pitch while stationary."""
        bias = self._imu.calibrate_gyro_bias()
        self._section_classifier.calibrate(self._imu.accel_pitch_baseline_rad)
        return bias

    def read(self):
        """Return one combined sensor sample and update landmark odometry."""
        now = monotonic()
        vehicle = self._vehicle.read()
        imu = self._imu.read()
        tof = self._tof.read()
        fuel = self._fuel_gauge.read()

        elapsed_s = max(now - self._last_sample_time, 0.001)
        self._last_sample_time = now
        section = self._section_classifier.update(imu, elapsed_s)
        step_m = vehicle["wheel_pulses"] * MM_PER_PULSE / 1000.0
        self.lap_distance_m += step_m
        self.total_distance_m += step_m
        self.speed_m_s = 0.8 * self.speed_m_s + 0.2 * (step_m / elapsed_s)

        new_landmark = False
        if tof is not None:
            new_landmark = self._tof.new_overhead(
                tof["distance_cm"], tof["range_status"]
            )
            if new_landmark:
                self.landmark_count += 1
                if self.landmark_count % 2 == 0:
                    self.lap_id += 1
                    self.lap_distance_m = 0.0

        sample = {}
        sample.update(vehicle)
        sample.update(imu)
        sample["fuel_gauge_voltage_v"] = fuel["voltage_v"]
        sample["fuel_percent"] = fuel["percent"]
        sample["lap_id"] = self.lap_id
        sample.update(section)
        sample["section_type"] = "{}_{}".format(
            section["turn_type"], section["grade_type"]
        )
        sample["distance_cm"] = tof["distance_cm"] if tof is not None else None
        sample["range_status"] = tof["range_status"] if tof is not None else None
        sample["new_landmark"] = new_landmark
        sample["landmark_count"] = self.landmark_count
        sample["lap_distance_m"] = self.lap_distance_m
        sample["total_distance_m"] = self.total_distance_m
        sample["speed_m_s"] = self.speed_m_s
        return sample

    def deinit(self):
        """Release the vehicle sensor hardware."""
        self._tof.deinit()
        self._vehicle.deinit()
