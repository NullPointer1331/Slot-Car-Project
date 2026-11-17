"""MicroPython CSV logger for a microSD card connected over SPI."""

import os

import sdcard

try:
    import machine
except ImportError:  # pragma: no cover - host builds do not use this module
    machine = None


class SDCardCSVLogger:
    """Mount a SPI microSD card and append rows to a CSV file."""

    def __init__(self, spi, chip_select_pin, filename, header):
        self._spi = spi
        self._filename = filename
        self._header = header
        if not filename:
            raise ValueError("filename must not be empty")
        if not header:
            raise ValueError("header must not be empty")
        if machine is None:
            raise RuntimeError("MicroPython's machine module is required")
        self._chip_select = machine.Pin(chip_select_pin, machine.Pin.OUT)
        self._chip_select.value(1)
        self._sd_card = sdcard.SDCard(spi, self._chip_select)
        self._filesystem = os.VfsFat(self._sd_card)
        self._mounted = False
        try:
            try:
                os.stat("/sd")
            except OSError as error:
                if not error.args or error.args[0] != 2:
                    raise
                os.mkdir("/sd")
            os.mount(self._filesystem, "/sd")
            self._mounted = True
            self._file = open(filename, "a")
            if os.stat(filename)[6] == 0:
                self._file.write(header)
                self._file.flush()
        except Exception:
            self._chip_select.value(1)
            raise

    def write_row(self, values):
        """Append one row, leaving unavailable values as empty CSV cells."""
        self._file.write(
            ",".join("" if value is None else str(value) for value in values) + "\n"
        )

    def flush(self):
        """Flush buffered CSV data to the card."""
        self._file.flush()

    def deinit(self):
        """Flush and close the CSV file, unmount the card, and release chip select."""
        try:
            self._file.flush()
        finally:
            try:
                self._file.close()
            finally:
                try:
                    if self._mounted:
                        os.umount("/sd")
                        self._mounted = False
                finally:
                    self._chip_select.value(1)
