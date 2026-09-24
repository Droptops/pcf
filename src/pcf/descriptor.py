"""Separate computation identity from a declared storage/transfer format."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace

from .segments import canonical_bytes
from .validation import digest, integer, nonempty, number


def sha256_tag(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_object(domain: str, value) -> str:
    return "sha256:" + hashlib.sha256(domain.encode() + b"\x00" + canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class Layout:
    positional: str
    dtype: str
    attention: str
    kv_quant: str | None = None
    block_tokens: int | None = None
    positional_config_hash: str | None = None
    format_id: str | None = None
    tensor_shape: tuple[int, ...] | None = None
    axis_order: tuple[str, ...] | None = None
    shard_spec_hash: str | None = None
    quantization_config_hash: str | None = None

    def __post_init__(self):
        for value, allowed in [(self.positional, {"rope", "alibi", "absolute", "nope"}),
                               (self.dtype, {"fp32", "bf16", "fp16", "fp8"}),
                               (self.attention, {"mha", "gqa", "mqa", "mla"}),
                               (self.kv_quant, {None, "int8", "int4", "fp8"})]:
            if value not in allowed:
                raise ValueError("unknown layout value")
        if self.block_tokens is not None:
            integer(self.block_tokens, "block_tokens", minimum=1)
        for name in ("positional_config_hash", "shard_spec_hash", "quantization_config_hash"):
            if getattr(self, name) is not None:
                digest(getattr(self, name), name)
        if self.format_id is not None:
            nonempty(self.format_id, "format_id")
        if self.tensor_shape is not None:
            object.__setattr__(self, "tensor_shape", tuple(self.tensor_shape))
            if not self.tensor_shape:
                raise ValueError("tensor_shape cannot be empty")
            for dim in self.tensor_shape:
                integer(dim, "tensor dimension", minimum=1)
        if self.axis_order is not None:
            object.__setattr__(self, "axis_order", tuple(self.axis_order))
            if not self.axis_order or len(set(self.axis_order)) != len(self.axis_order):
                raise ValueError("axis_order must name unique axes")
            for axis in self.axis_order:
                nonempty(axis, "axis")
        if self.tensor_shape and self.axis_order and len(self.tensor_shape) != len(self.axis_order):
            raise ValueError("shape and axis order differ in rank")

    @property
    def complete(self) -> bool:
        return all((self.format_id, self.tensor_shape, self.axis_order, self.block_tokens,
                    self.shard_spec_hash, self.positional_config_hash)) and (
                        self.kv_quant is None or self.quantization_config_hash is not None)


@dataclass(frozen=True)
class Engine:
    name: str
    version: str

    def __post_init__(self):
        nonempty(self.name, "engine name")
        nonempty(self.version, "engine version")


@dataclass(frozen=True)
class CacheDescriptor:
    family: str
    model_id: str
    weights_hash: str
    tokenizer_hash: str
    layout: Layout
    min_cacheable_tokens: int
    max_breakpoints: int
    ttl_seconds: int
    engine: Engine | None = None
    descriptor_version: str = "0.2"
    identity_kind: str = "opaque"
    execution_config_hash: str | None = None
    cache_write_multiplier: float = 1.25

    def __post_init__(self):
        nonempty(self.family, "family")
        nonempty(self.model_id, "model_id")
        digest(self.weights_hash, "weights_hash")
        digest(self.tokenizer_hash, "tokenizer_hash")
        if self.identity_kind not in {"opaque", "simulated", "manifest"}:
            raise ValueError("identity_kind must be opaque, simulated or manifest")
        if self.descriptor_version != "0.2":
            raise ValueError("unsupported descriptor version")
        integer(self.min_cacheable_tokens, "min_cacheable_tokens")
        integer(self.max_breakpoints, "max_breakpoints")
        integer(self.ttl_seconds, "ttl_seconds", minimum=1)
        number(self.cache_write_multiplier, "cache_write_multiplier")
        if self.execution_config_hash is not None:
            digest(self.execution_config_hash, "execution_config_hash")
        if self.identity_kind == "manifest" and not self.execution_config_hash:
            raise ValueError("manifest identity requires execution_config_hash")

    @property
    def compat_key(self) -> str:
        """Computation grouping only. Never independently authorizes raw KV exchange."""
        return hash_object("pcf:execution:0.2", {
            "identity_kind": self.identity_kind, "weights_hash": self.weights_hash,
            "tokenizer_hash": self.tokenizer_hash, "execution_config_hash": self.execution_config_hash,
            "positional": self.layout.positional, "positional_config_hash": self.layout.positional_config_hash,
            "dtype": self.layout.dtype, "kv_quant": self.layout.kv_quant,
            "quantization_config_hash": self.layout.quantization_config_hash, "attention": self.layout.attention})

    @property
    def byte_compat_key(self) -> str | None:
        """Declared ABI identity; absent for opaque/simulated/incomplete descriptors.

        The caller must verify manifest truth and transport authorization. No KV
        byte transport or authenticity attestation is implemented by this package.
        """
        if self.identity_kind != "manifest" or not self.layout.complete:
            return None
        return hash_object("pcf:storage:0.2", {"execution": self.compat_key, "layout": asdict(self.layout)})

    def byte_compatible_with(self, other: CacheDescriptor) -> bool:
        return self.byte_compat_key is not None and self.byte_compat_key == other.byte_compat_key

    def with_engine(self, name: str, version: str, block_tokens: int | None = None) -> CacheDescriptor:
        layout = self.layout if block_tokens is None else replace(self.layout, block_tokens=block_tokens)
        return replace(self, engine=Engine(name, version), layout=layout)

    def to_json(self) -> dict:
        # Normalize tuple fields into JSON arrays.
        import json
        result = json.loads(canonical_bytes(asdict(self)))
        result["compat_key"] = self.compat_key
        result["byte_compat_key"] = self.byte_compat_key
        return result
