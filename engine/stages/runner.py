"""Runs a sequence of stages over a shared state and records what each one did."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


class StageParams:
    """Render parameters that record which stage reads which key.

    A key nobody reads is a parameter that has no effect on this render; the
    trace reports those.
    """

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values: Dict[str, Any] = dict(values)
        self._reads: Dict[str, List[str]] = {}
        self._sampled: Dict[str, Any] = {}
        self._reader: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        self._record(key)
        return self._values[key]

    def get(self, key: str, default: Any = None) -> Any:
        self._record(key)
        return self._values.get(key, default)

    def get_or_sample(self, key: str, sample: Callable[[], Any]) -> Any:
        """Returns the value, sampling it once on first read if it is None."""
        self._record(key)
        value = self._values.get(key)
        if value is None:
            value = sample()
            self._values[key] = value
            self._sampled[key] = value
        return value

    def keys(self) -> Tuple[str, ...]:
        return tuple(self._values)

    def reads(self, reader: str) -> Tuple[str, ...]:
        return tuple(self._reads.get(reader, ()))

    def unread(self) -> Tuple[str, ...]:
        read = {key for keys in self._reads.values() for key in keys}
        return tuple(key for key in self._values if key not in read)

    @property
    def sampled(self) -> Dict[str, Any]:
        return dict(self._sampled)

    @contextmanager
    def reading_for(self, reader: str) -> Iterator[None]:
        previous, self._reader = self._reader, reader
        try:
            yield
        finally:
            self._reader = previous

    def _record(self, key: str) -> None:
        if self._reader is None:
            return
        keys = self._reads.setdefault(self._reader, [])
        if key not in keys:
            keys.append(key)


def _no_args(_: StageParams) -> Dict[str, Any]:
    return {}


@dataclass(frozen=True)
class Stage:
    """One step of a render.

    fn receives the inputs from the state positionally, in the declared order,
    plus the keyword arguments bind() derives from the render parameters; its
    return value is stored under output.
    """

    name: str
    phase: str  # palette, frame, warp, field, mapping, export
    fn: Callable[..., Any]
    inputs: Tuple[str, ...]
    output: str
    bind: Callable[[StageParams], Dict[str, Any]] = _no_args

    @property
    def uses_rng(self) -> bool:
        return "rng" in inspect.signature(self.fn).parameters


@dataclass
class StageRecord:
    """One executed step: what it read, what it was given, what it produced."""

    name: str
    phase: str
    inputs: Tuple[str, ...]
    output_name: str
    params_read: Tuple[str, ...]
    arguments: Dict[str, Any]
    output: Any


@dataclass
class Trace:
    """What a render did, stage by stage."""

    records: List[StageRecord] = field(default_factory=list)
    param_sets: List[StageParams] = field(default_factory=list)

    def watch(self, params: StageParams) -> None:
        """Includes params in the unread-parameter report."""
        if not any(existing is params for existing in self.param_sets):
            self.param_sets.append(params)

    def add(
        self,
        name: str,
        phase: str,
        output: Any,
        *,
        inputs: Tuple[str, ...] = (),
        output_name: str = "",
        arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records a step that ran outside run_stages; its arguments are its params."""
        arguments = dict(arguments or {})
        self.records.append(
            StageRecord(name, phase, inputs, output_name or name, tuple(arguments), arguments, output)
        )

    def unread_params(self) -> Tuple[str, ...]:
        return tuple(self.unread_values())

    def unread_values(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for params in self.param_sets:
            values.update({key: params.get(key) for key in params.unread()})
        return values


def run_stages(
    stages: Sequence[Stage],
    params: StageParams,
    *,
    rng: Any,
    state: Optional[Dict[str, Any]] = None,
    trace: Optional[Trace] = None,
) -> Dict[str, Any]:
    """Runs stages in order and returns the final state."""
    state = dict(state or {})
    if trace is not None:
        trace.watch(params)
    for stage in stages:
        with params.reading_for(stage.name):
            arguments = stage.bind(params)
        inputs = [state[name] for name in stage.inputs]
        extra = {"rng": rng} if stage.uses_rng else {}
        state[stage.output] = stage.fn(*inputs, **arguments, **extra)
        if trace is not None:
            trace.records.append(
                StageRecord(
                    stage.name,
                    stage.phase,
                    stage.inputs,
                    stage.output,
                    params.reads(stage.name),
                    arguments,
                    state[stage.output],
                )
            )
    return state
