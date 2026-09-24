"""Packaged JSON schemas, also mirrored under spec/schemas for implementers."""
import json
from functools import lru_cache
from importlib.resources import files

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


@lru_cache(maxsize=3)
def _validator(name):
    if name not in {"pcf", "cache-descriptor", "route-decision"}:
        raise ValueError("unknown PCF schema")
    schema = json.loads(files(__package__).joinpath(f"{name}.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate(name, value):
    try:
        _validator(name).validate(value)
    except ValidationError as exc:
        location = ".".join(str(p) for p in exc.absolute_path) or "root"
        raise ValueError(f"{name} validation failed at {location}: {exc.message}") from exc
