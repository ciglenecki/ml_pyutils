import os
from pathlib import Path
from typing import Any, Literal

from pydantic import SecretStr, ValidationError, field_validator, validator
from pydantic_settings import BaseSettings

from ml_pyutils.meta.singleton import Singleton


class InvalidEnvironmentVariable(Exception):
    pass


ENV_VALUE_TYPE = Literal["dev", "staging", "prod"]

SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 60 * SECONDS_IN_MINUTE


class PydanticConfig(BaseSettings):
    # SELF
    ENV: ENV_VALUE_TYPE
    SERVICE_NAME: Literal["<INSERT_NAME>"] = "<INSERT_NAME>"

    @field_validator("*", mode="after")
    def empty_str_to_none(cls, v):
        if v == "":
            return None
        return v

    @field_validator("device", mode="after")
    @classmethod
    def _validate_torch_device_str(cls, v: Any) -> str:
        import torch

        if not isinstance(v, str):
            raise TypeError(f"device field must be a string, got {type(v).__name__!r}")

        try:
            dev = torch.device(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid torch device string {v!r}: {e}") from e

        # enforce CUDA index bounds
        if dev.type == "cuda" and dev.index is not None:
            n = torch.cuda.device_count()
            if not (0 <= dev.index < n):
                raise ValueError(
                    f"cuda device index {dev.index} out of range (0..{n - 1})"
                )

        return v

    @field_validator("input_size", mode="before")
    @classmethod
    def parse_input_size(cls, v):
        """Parse input_size from comma-separated strings to tuples."""
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str):
                    # Split by comma and convert to integers
                    try:
                        tuple_values = tuple(int(x.strip()) for x in item.split(","))
                        result.append(tuple_values)
                    except (ValueError, AttributeError) as e:
                        raise ValueError(
                            f"Invalid input_size format: {item}. Expected comma-separated integers."
                        ) from e
                elif isinstance(item, (list, tuple)):
                    # Already a sequence, convert to tuple
                    result.append(tuple(item))
                else:
                    raise ValueError(
                        f"Invalid input_size item type: {type(item)}. Expected string or sequence."
                    )
            return result
        elif isinstance(v, str):
            # Single string, convert to list with one tuple
            tuple_values = tuple(int(x.strip()) for x in v.split(","))
            return [tuple_values]
        else:
            raise ValueError(
                f"Invalid input_size type: {type(v)}. Expected list or string."
            )

    def model_post_init(self, __context) -> None:
        pass


def loc_to_dot_sep(loc: tuple[str | int, ...]) -> str:
    path = ""
    for i, x in enumerate(loc):
        if isinstance(x, str):
            if i > 0:
                path += "."
            path += x
        elif isinstance(x, int):
            path += f"[{x}]"
        else:
            raise TypeError("Unexpected type")
    return path


class ConfigSingleton(metaclass=Singleton):
    config: PydanticConfig

    def __init__(self, env_files: list[Path] | None = None) -> None:
        """Create env files from `env_files` argument and ENV_FILES environment variable if it exists.

        Priority from highest to lowest:
            variables from env
            env_files argument
            ENV_FILES environment variable
        """
        env_string = os.getenv("ENV_FILES")

        if env_string:
            env_files_from_env = [Path(env_path) for env_path in env_string.split(":")]
            if env_files is not None:
                env_files_from_env.extend(env_files)
            env_files = env_files_from_env

        if env_files is None:
            env_files = [Path(".env")]

        try:
            self.config = PydanticConfig(
                _env_file=env_files,  # type: ignore
                _env_file_encoding="utf-8",  # type: ignore
            )
            print(self.config.model_dump_json())
        except ValidationError as e:
            bad_env_vars = [
                f"{loc_to_dot_sep(e['loc'])}: {e['type']}" for e in e.errors()
            ]
            bad_env_str = "\n".join(bad_env_vars)
            msg = (
                f"\n\nInvalid enviroment variables. "
                "You can set ENV variables directly or use ENV_FILES env to pass a list of .env files to load.\n\n"
                f"{bad_env_str}"
            )
            raise InvalidEnvironmentVariable(msg) from e


if __name__ == "__main__":
    ConfigSingleton()
