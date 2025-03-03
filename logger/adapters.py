from __future__ import annotations

import json
import logging
import logging.config
from abc import ABC, abstractmethod

import shortuuid
from rq import get_current_job

LoggerType = logging.Logger | logging.LoggerAdapter


class ExtraToJsonAdapter(logging.LoggerAdapter, ABC):
    """
    Converts all extra fields to json string and puts them under `extra` key.

    Before: extra={"a": 1, "b": 2}
    After:  extra={"extra": "{\"a\": 1, \"b\": 2}"}
    Usage: log.info("Some message", extra={"any_key": "any_value"})

    warning: if you want to add extra fields which won't be converted to json string,
    you have to add them before this adapter.

    Problem this class solves:
        You don' know concrete extra fields that you want to add to the log message.
        Even if you do, you would have to add them to every log call.
        Extra fields are usually added to the formatter.
        format: "%(asctime)s ... %(my_field)s %(another_field)s"

    By always using a single extra field called "extra"
    with the following format: "%(asctime)s ... %(extra)s"
    you can add any extra fields to the log message by passing them in the extra argument.
    """

    def process(self, msg, kwargs):
        extra = kwargs.get("extra")
        if extra is None:
            extra = {}

        extra.pop(None, None)  # pop None values

        json_str = json.dumps(extra, sort_keys=True, default=str)
        extra = dict(extra=json_str)
        kwargs["extra"] = extra
        return msg, kwargs


class AppendExtraAdapter(logging.LoggerAdapter, ABC):
    """
    This adapter class is used to add extra fields to the log's extra dict.
    Concrete classes have to implement the get_extra which returns a dict.

    Before: extra = {"a": 1, "b": 2} | self.get_extra()
    After:  extra = {"a": 1, "b": 2, "c": 3, "d": 4}
    """

    def process(self, msg, kwargs):
        extra = kwargs.get("extra")
        if extra is None:
            extra = {}
        extra = self.get_extra(msg, kwargs) | extra
        kwargs["extra"] = extra
        return msg, kwargs

    @abstractmethod
    def get_extra(self, msg, kwargs) -> dict:
        raise NotImplementedError


class PermanentExtraAdapter(AppendExtraAdapter):
    """
    Appends an input dictionary (passed as `extra`) to the log's extra.

    Usage:
        PermanentExtraAdapter(extra={
            "logger_id": "98q39",
            "any_key": "blabla"
        })

    Before: extra = {}
    After:  extra = {"logger_id": "98q39", "any_key": "blabla"}
    """

    def __init__(self, logger: LoggerType, extra: dict):
        if not isinstance(extra, dict):
            raise ValueError("extra has to be a dictionary")
        super().__init__(logger, extra)

    def get_extra(self, msg, kwargs):
        return dict(self.extra)  # type: ignore


class ExceptionLoggingAdapter(AppendExtraAdapter):
    """
    If exception exists, adds error fields to the extra of the log message.
    Exception has to be passed explicitly in the exc_info argument.

    Before: extra = {}
    After:  extra = {
                "error_type": "ValueError",
                "error_value": "Some error",
                "error_id":"98q39"
            }
    """

    def get_error_dict(self, exc_info: BaseException | tuple):
        if isinstance(exc_info, BaseException):
            _, error_value = type(exc_info), exc_info
        elif isinstance(exc_info, tuple):
            _, error_value, _ = exc_info

        return dict(
            error_type=type(error_value).__name__,
            error_value=error_value,
            error_id=shortuuid.uuid(),
        )

    def get_extra(self, msg, kwargs):
        exc_info = kwargs.get("exc_info")
        if not exc_info:
            return {}
        return self.get_error_dict(exc_info)
