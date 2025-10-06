"""
Example Usage:

from human_bytes import HumanBytes

HumanBytes.format(1500) # '1.5 KB'
HumanBytes.format(1536, metric=False) # '1.5 KiB'
HumanBytes.format(5_368_709_120, precision=2) # '5.37 GB'
HumanBytes.format(1024, metric=False) # '1 KiB'
HumanBytes.format(1234.56) # '1.2 KB'
"""

class HumanBytes:
    """
    Utility class to format bytes into human readable strings.
    Uses metric (SI) units by default (KB, MB, GB) rather than binary units (KiB, MiB, GiB).
    """

    METRIC_LABELS = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    BINARY_LABELS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]
    METRIC_UNIT = 1000.0
    BINARY_UNIT = 1024.0

    @staticmethod
    def format(num_bytes: int | float, metric: bool = True, precision: int = 1) -> str:
        """
        Format bytes into human readable string.

        Args:
            num_bytes: Number of bytes to format
            metric: If True, use metric (SI) units (KB, MB, GB), else use binary units (KiB, MiB, GiB)
            precision: Number of decimal places to show

        Returns:
            Formatted string like "1.5 MB" or "2 GiB"
        """
        unit = HumanBytes.METRIC_UNIT if metric else HumanBytes.BINARY_UNIT
        labels = HumanBytes.METRIC_LABELS if metric else HumanBytes.BINARY_LABELS

        if num_bytes < unit:
            return f"{num_bytes} {labels[0]}"

        exponent = min(int(math.log(num_bytes, unit)), len(labels) - 1)
        quotient = float(num_bytes) / (unit**exponent)

        if quotient.is_integer():
            precision = 0

        return f"{quotient:.{precision}f} {labels[exponent]}"
