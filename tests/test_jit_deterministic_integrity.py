from __future__ import annotations

import pytest

from pcf.jit.deterministic import (
    DeterministicArtifact,
    DeterministicExecutionError,
    compile_deterministic_artifact,
    execute_deterministic_artifact,
)


def test_artifact_constructor_is_not_a_public_bypass() -> None:
    with pytest.raises(TypeError, match="compile_deterministic_artifact"):
        DeterministicArtifact()


def test_unsealed_artifact_fails_closed_in_direct_executor() -> None:
    unsealed = object.__new__(DeterministicArtifact)
    with pytest.raises(DeterministicExecutionError, match="produced by compile"):
        execute_deterministic_artifact(unsealed, {})


def test_composite_constant_output_cannot_mutate_compiled_program() -> None:
    artifact = compile_deterministic_artifact(
        name="constant_payload",
        version="1",
        expression={"op": "const", "value": {"items": [1, 2]}},
    )

    first = execute_deterministic_artifact(artifact, {})
    first["items"].append(3)

    assert first == {"items": [1, 2, 3]}
    assert execute_deterministic_artifact(artifact, {}) == {"items": [1, 2]}
