"""CSV logging to a microSD card connected over SPI."""

import os

import adafruit_sdcard
import digitalio
import storage


class SDCardCSVLogger:
    """Mount a microSD card and append rows to a CSV file."""

    def __init__(self, spi, chip_select_pin, filename, header):
        self._chip_select = digitalio.DigitalInOut(chip_select_pin)
        self._sd_card = adafruit_sdcard.SDCard(spi, self._chip_select)
        self._filesystem = storage.VfsFat(self._sd_card)
        storage.mount(self._filesystem, "/sd")
        self._file = open(filename, "a")

        if os.stat(filename)[6] == 0:
            self._file.write(header)
            self._file.flush()

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
        self._file.flush()
        self._file.close()
        storage.umount("/sd")
        self._chip_select.deinit()
