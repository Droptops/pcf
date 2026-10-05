from __future__ import annotations

import math

import pytest

from pcf.jit.deterministic import (
    DeterministicCompileError,
    DeterministicExecutionError,
    DeterministicPolicy,
    compile_deterministic_artifact,
    execute_deterministic_artifact,
)


def test_compiles_and_executes_bounded_formula() -> None:
    artifact = compile_deterministic_artifact(
        name="celsius_to_fahrenheit",
        version="1",
        dependencies={"formula": "v1"},
        expression={
            "op": "add",
            "args": [
                {
                    "op": "mul",
                    "args": [
                        {"op": "input", "path": "temperature_c"},
                        {"op": "const", "value": 1.8},
                    ],
                },
                {"op": "const", "value": 32},
            ],
        },
    )

    assert execute_deterministic_artifact(artifact, {"temperature_c": 37}) == pytest.approx(98.6)
    assert artifact.node_count == 5
    assert len(artifact.artifact_sha256) == 64


def test_object_projection_and_string_normalization() -> None:
    artifact = compile_deterministic_artifact(
        name="normalize_customer",
        version="1",
        expression={
            "op": "object",
            "fields": {
                "customer_id": {"op": "input", "path": "customer.id"},
                "email": {
                    "op": "lower",
                    "arg": {"op": "strip", "arg": {"op": "input", "path": "customer.email"}},
                },
            },
        },
    )

    result = execute_deterministic_artifact(
        artifact,
        {"customer": {"id": "c-123", "email": "  PERSON@EXAMPLE.COM "}},
    )
    assert result == {"customer_id": "c-123", "email": "person@example.com"}


def test_conditionals_require_boolean_condition() -> None:
    artifact = compile_deterministic_artifact(
        name="bucket",
        version="1",
        expression={
            "op": "if",
            "condition": {
                "op": "gte",
                "args": [{"op": "input", "path": "score"}, {"op": "const", "value": 80}],
            },
            "then": {"op": "const", "value": "high"},
            "else": {"op": "const", "value": "low"},
        },
    )

    assert execute_deterministic_artifact(artifact, {"score": 91}) == "high"
    assert execute_deterministic_artifact(artifact, {"score": 12}) == "low"


def test_artifact_hash_is_stable_across_dependency_order() -> None:
    expression = {"op": "input", "path": "value"}
    left = compile_deterministic_artifact(
        name="stable",
        version="1",
        expression=expression,
        dependencies={"b": "2", "a": "1"},
    )
    right = compile_deterministic_artifact(
        name="stable",
        version="1",
        expression=expression,
        dependencies={"a": "1", "b": "2"},
    )
    assert left.artifact_sha256 == right.artifact_sha256


def test_rejects_unknown_operation_instead_of_evaluating_source() -> None:
    with pytest.raises(DeterministicCompileError, match="unsupported operation"):
        compile_deterministic_artifact(
            name="unsafe",
            version="1",
            expression={"op": "eval", "args": [{"op": "const", "value": "__import__('os')"}]},
        )


def test_rejects_attribute_style_escape_paths() -> None:
    with pytest.raises(DeterministicCompileError, match="forbidden component"):
        compile_deterministic_artifact(
            name="unsafe_path",
            version="1",
            expression={"op": "input", "path": "payload.__class__"},
        )


def test_missing_input_and_division_by_zero_fail_closed() -> None:
    missing = compile_deterministic_artifact(
        name="missing",
        version="1",
        expression={"op": "input", "path": "a.b"},
    )
    with pytest.raises(DeterministicExecutionError, match="missing input path"):
        execute_deterministic_artifact(missing, {"a": {}})

    divide = compile_deterministic_artifact(
        name="divide",
        version="1",
        expression={
            "op": "div",
            "args": [{"op": "input", "path": "x"}, {"op": "input", "path": "y"}],
        },
    )
    with pytest.raises(DeterministicExecutionError, match="division by zero"):
        execute_deterministic_artifact(divide, {"x": 1, "y": 0})


def test_rejects_non_finite_inputs_and_constants() -> None:
    with pytest.raises(DeterministicCompileError, match="non-finite"):
        compile_deterministic_artifact(
            name="nan",
            version="1",
            expression={"op": "const", "value": math.nan},
        )

    artifact = compile_deterministic_artifact(
        name="input",
        version="1",
        expression={"op": "input", "path": "value"},
    )
    with pytest.raises(DeterministicExecutionError, match="non-finite"):
        execute_deterministic_artifact(artifact, {"value": math.inf})


def test_compile_limits_are_enforced() -> None:
    expression = {
        "op": "add",
        "args": [
            {"op": "const", "value": 1},
            {"op": "const", "value": 2},
            {"op": "const", "value": 3},
        ],
    }
    with pytest.raises(DeterministicCompileError, match="maximum node count"):
        compile_deterministic_artifact(
            name="too_large",
            version="1",
            expression=expression,
            policy=DeterministicPolicy(max_nodes=3),
        )


def test_numeric_operators_do_not_accept_booleans() -> None:
    artifact = compile_deterministic_artifact(
        name="bool_math",
        version="1",
        expression={
            "op": "add",
            "args": [{"op": "input", "path": "x"}, {"op": "const", "value": 1}],
        },
    )
    with pytest.raises(DeterministicExecutionError, match="finite numeric operands"):
        execute_deterministic_artifact(artifact, {"x": True})
