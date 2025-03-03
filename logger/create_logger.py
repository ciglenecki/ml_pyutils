import logging
import logging.config
from pathlib import Path
from typing import cast

import shortuuid

from logger.adapters import (
    ExceptionLoggingAdapter,
    ExtraToJsonAdapter,
    PermanentExtraAdapter,
)

SERVICE_NAME = "my_service_name"
LOG_FILE = Path("logs") / f"{SERVICE_NAME}.log"
LOG_FILE_REFRESH_DAY = 1
LOG_FILE_BACKUP_COUNT = 130
MAX_LOGGER_ID_LENGTH = 6
LOG_LEVEL = "INFO"
LOG_HEADER_LINE_DELIMITER = " - "
FIRST_LINE_DELIMITER = "\u200b"  # zero width space, https://grafana.com/docs/loki/latest/send-data/promtail/stages/multiline/#custom-log-format


def get_logger_format(
    header_line_delimiter: str,
    service_name: str,
) -> str:
    """
    Example of format: "%(asctime)s - %(levelname)s - (service_name) - %(message)s\n%(extra)s"
    """
    header_line = header_line_delimiter.join(
        [
            "%(asctime)s",
            "%(levelname)s",
            service_name,
            "%(logger_id)s",
            "%(message)s",
        ]
    )
    extra_line = "%(extra)s"
    format_string = f"{header_line}\n{extra_line}\n"

    return format_string


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": FIRST_LINE_DELIMITER
            + get_logger_format(
                header_line_delimiter=LOG_HEADER_LINE_DELIMITER,
                service_name=SERVICE_NAME,
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "fileHandler": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "standard",
            "filename": str(Path(LOG_LEVEL)),
            "when": "D",  # day
            "interval": LOG_FILE_REFRESH_DAY,
            "backupCount": LOG_FILE_BACKUP_COUNT,
            "level": LOG_LEVEL,
        },
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": LOG_LEVEL,
        },
    },
    "loggers": {
        SERVICE_NAME: {
            "handlers": ["fileHandler", "consoleHandler"],
            "propagate": False,
            "level": LOG_LEVEL,
        },
    },
}


def create_logger(
    log_file: Path = LOG_FILE,
    logger_name: str = SERVICE_NAME,
    logger_id_length: int = MAX_LOGGER_ID_LENGTH,
) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
    __log = logging.getLogger(logger_name)
    __log = PermanentExtraAdapter(
        __log, extra=dict(logger_id=shortuuid.uuid()[:logger_id_length])
    )
    # everything bellow this adapter will be turned to json
    __log = ExtraToJsonAdapter(__log, __log.extra)
    __log = ExceptionLoggingAdapter(__log, __log.extra)
    log = cast(logging.Logger, __log)
    return log
