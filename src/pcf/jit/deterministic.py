"""Restricted deterministic execution target for audited JIT hot paths.

The runtime executes a small declarative expression language. It never evaluates
source code, imports modules, performs I/O, or resolves object attributes. Inputs
and outputs are finite JSON-like values only.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence


class DeterministicCompileError(ValueError):
    """The declarative program is invalid or exceeds compiler limits."""


class DeterministicExecutionError(RuntimeError):
    """The compiled program could not execute deterministically."""


_ALLOWED_OPS = frozenset(
    {
        "input",
        "const",
        "object",
        "array",
        "add",
        "sub",
        "mul",
        "div",
        "eq",
        "ne",
        "lt",
        "lte",
        "gt",
        "gte",
        "and",
        "or",
        "not",
        "if",
        "concat",
        "lower",
        "upper",
        "strip",
    }
)


@dataclass(frozen=True)
class DeterministicPolicy:
    max_nodes: int = 256
    max_depth: int = 32
    max_string_length: int = 100_000
    max_collection_items: int = 1_000

    def __post_init__(self) -> None:
        for name in ("max_nodes", "max_depth", "max_string_length", "max_collection_items"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DeterministicArtifact:
    name: str
    version: str
    expression: Mapping[str, Any]
    dependencies: tuple[tuple[str, str], ...]
    artifact_sha256: str
    node_count: int
    max_depth: int


def _finite_json(value: Any, *, policy: DeterministicPolicy, depth: int = 0) -> None:
    if depth > policy.max_depth:
        raise DeterministicCompileError("JSON value exceeds maximum depth")
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DeterministicCompileError("non-finite numbers are not allowed")
        return
    if isinstance(value, str):
        if len(value) > policy.max_string_length:
            raise DeterministicCompileError("string exceeds maximum length")
        return
    if isinstance(value, Mapping):
        if len(value) > policy.max_collection_items:
            raise DeterministicCompileError("object exceeds maximum item count")
        for key, item in value.items():
            if not isinstance(key, str):
                raise DeterministicCompileError("object keys must be strings")
            _finite_json(item, policy=policy, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > policy.max_collection_items:
            raise DeterministicCompileError("array exceeds maximum item count")
        for item in value:
            _finite_json(item, policy=policy, depth=depth + 1)
        return
    raise DeterministicCompileError(f"unsupported value type: {type(value).__name__}")


def _walk_expression(expr: Any, *, policy: DeterministicPolicy, depth: int = 1) -> tuple[int, int]:
    if depth > policy.max_depth:
        raise DeterministicCompileError("expression exceeds maximum depth")
    if not isinstance(expr, Mapping):
        raise DeterministicCompileError("each expression node must be an object")
    op = expr.get("op")
    if op not in _ALLOWED_OPS:
        raise DeterministicCompileError(f"unsupported operation: {op!r}")

    keys = set(expr)
    children: list[Any] = []
    if op == "input":
        if keys != {"op", "path"} or not isinstance(expr.get("path"), str) or not expr["path"].strip():
            raise DeterministicCompileError("input requires a non-empty string path")
        if any(part in {"", "__class__", "__dict__", "__globals__"} for part in expr["path"].split(".")):
            raise DeterministicCompileError("input path contains a forbidden component")
    elif op == "const":
        if keys != {"op", "value"}:
            raise DeterministicCompileError("const requires exactly op and value")
        _finite_json(expr["value"], policy=policy)
    elif op == "object":
        if keys != {"op", "fields"} or not isinstance(expr.get("fields"), Mapping):
            raise DeterministicCompileError("object requires a fields object")
        if len(expr["fields"]) > policy.max_collection_items:
            raise DeterministicCompileError("object expression exceeds maximum item count")
        for key, child in expr["fields"].items():
            if not isinstance(key, str) or not key:
                raise DeterministicCompileError("object expression keys must be non-empty strings")
            children.append(child)
    elif op == "array":
        if keys != {"op", "items"} or not isinstance(expr.get("items"), Sequence) or isinstance(expr["items"], (str, bytes)):
            raise DeterministicCompileError("array requires an items array")
        if len(expr["items"]) > policy.max_collection_items:
            raise DeterministicCompileError("array expression exceeds maximum item count")
        children.extend(expr["items"])
    elif op in {"not", "lower", "upper", "strip"}:
        if keys != {"op", "arg"}:
            raise DeterministicCompileError(f"{op} requires exactly one arg")
        children.append(expr["arg"])
    elif op == "if":
        if keys != {"op", "condition", "then", "else"}:
            raise DeterministicCompileError("if requires condition, then, and else")
        children.extend((expr["condition"], expr["then"], expr["else"] ))
    else:
        if keys != {"op", "args"} or not isinstance(expr.get("args"), Sequence) or isinstance(expr["args"], (str, bytes)):
            raise DeterministicCompileError(f"{op} requires an args array")
        args = list(expr["args"])
        expected = 2 if op in {"sub", "div", "eq", "ne", "lt", "lte", "gt", "gte"} else None
        if expected is not None and len(args) != expected:
            raise DeterministicCompileError(f"{op} requires exactly {expected} args")
        if op in {"add", "mul", "and", "or", "concat"} and not args:
            raise DeterministicCompileError(f"{op} requires at least one arg")
        children.extend(args)

    total = 1
    max_depth = depth
    for child in children:
        child_nodes, child_depth = _walk_expression(child, policy=policy, depth=depth + 1)
        total += child_nodes
        max_depth = max(max_depth, child_depth)
        if total > policy.max_nodes:
            raise DeterministicCompileError("expression exceeds maximum node count")
    return total, max_depth


def compile_deterministic_artifact(
    *,
    name: str,
    version: str,
    expression: Mapping[str, Any],
    dependencies: Mapping[str, str] | None = None,
    policy: DeterministicPolicy | None = None,
) -> DeterministicArtifact:
    """Validate and freeze a deterministic artifact with a stable content hash."""
    policy = policy or DeterministicPolicy()
    if not isinstance(name, str) or not name.strip():
        raise DeterministicCompileError("name must be non-empty")
    if not isinstance(version, str) or not version.strip():
        raise DeterministicCompileError("version must be non-empty")
    deps = dependencies or {}
    if not isinstance(deps, Mapping):
        raise DeterministicCompileError("dependencies must be a mapping")
    normalized_deps: list[tuple[str, str]] = []
    for key, value in deps.items():
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise DeterministicCompileError("dependency names and versions must be non-empty strings")
        normalized_deps.append((key, value))
    normalized_deps.sort()
    node_count, max_depth = _walk_expression(expression, policy=policy)
    payload = {
        "schema": "pcf.deterministic.v1",
        "name": name,
        "version": version,
        "dependencies": normalized_deps,
        "expression": expression,
    }
    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise DeterministicCompileError("artifact is not canonical JSON") from exc
    return DeterministicArtifact(
        name=name,
        version=version,
        expression=dict(expression),
        dependencies=tuple(normalized_deps),
        artifact_sha256=sha256(canonical).hexdigest(),
        node_count=node_count,
        max_depth=max_depth,
    )


def _numeric(value: Any, op: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DeterministicExecutionError(f"{op} requires finite numeric operands")
    return value


def _resolve_input(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise DeterministicExecutionError(f"missing input path: {path}")
        current = current[part]
    return current


def _eval(expr: Mapping[str, Any], data: Mapping[str, Any], policy: DeterministicPolicy, depth: int = 1) -> Any:
    if depth > policy.max_depth:
        raise DeterministicExecutionError("execution exceeded maximum depth")
    op = expr["op"]
    if op == "input":
        return _resolve_input(data, expr["path"])
    if op == "const":
        return expr["value"]
    if op == "object":
        return {key: _eval(value, data, policy, depth + 1) for key, value in expr["fields"].items()}
    if op == "array":
        return [_eval(value, data, policy, depth + 1) for value in expr["items"]]
    if op == "if":
        condition = _eval(expr["condition"], data, policy, depth + 1)
        if not isinstance(condition, bool):
            raise DeterministicExecutionError("if condition must evaluate to bool")
        branch = expr["then"] if condition else expr["else"]
        return _eval(branch, data, policy, depth + 1)
    if op in {"not", "lower", "upper", "strip"}:
        value = _eval(expr["arg"], data, policy, depth + 1)
        if op == "not":
            if not isinstance(value, bool):
                raise DeterministicExecutionError("not requires a bool operand")
            return not value
        if not isinstance(value, str):
            raise DeterministicExecutionError(f"{op} requires a string operand")
        return {"lower": str.lower, "upper": str.upper, "strip": str.strip}[op](value)

    args = [_eval(item, data, policy, depth + 1) for item in expr["args"]]
    if op == "concat":
        if not all(isinstance(item, str) for item in args):
            raise DeterministicExecutionError("concat requires string operands")
        return "".join(args)
    if op in {"and", "or"}:
        if not all(isinstance(item, bool) for item in args):
            raise DeterministicExecutionError(f"{op} requires bool operands")
        return all(args) if op == "and" else any(args)
    if op in {"eq", "ne"}:
        return (args[0] == args[1]) if op == "eq" else (args[0] != args[1])
    if op in {"lt", "lte", "gt", "gte"}:
        left = _numeric(args[0], op)
        right = _numeric(args[1], op)
        return {"lt": left < right, "lte": left <= right, "gt": left > right, "gte": left >= right}[op]
    if op in {"add", "sub", "mul", "div"}:
        nums = [_numeric(item, op) for item in args]
        if op == "add":
            return sum(nums)
        if op == "sub":
            return nums[0] - nums[1]
        if op == "mul":
            result: int | float = 1
            for item in nums:
                result *= item
            return result
        if nums[1] == 0:
            raise DeterministicExecutionError("division by zero")
        return nums[0] / nums[1]
    raise DeterministicExecutionError(f"unsupported compiled operation: {op}")


def execute_deterministic_artifact(
    artifact: DeterministicArtifact,
    inputs: Mapping[str, Any],
    *,
    policy: DeterministicPolicy | None = None,
) -> Any:
    """Execute a validated artifact over finite JSON-like inputs."""
    if not isinstance(artifact, DeterministicArtifact):
        raise DeterministicExecutionError("artifact must be DeterministicArtifact")
    if not isinstance(inputs, Mapping):
        raise DeterministicExecutionError("inputs must be a mapping")
    policy = policy or DeterministicPolicy()
    try:
        _finite_json(inputs, policy=policy)
    except DeterministicCompileError as exc:
        raise DeterministicExecutionError(str(exc)) from exc
    result = _eval(artifact.expression, inputs, policy)
    try:
        _finite_json(result, policy=policy)
    except DeterministicCompileError as exc:
        raise DeterministicExecutionError(str(exc)) from exc
    return result
