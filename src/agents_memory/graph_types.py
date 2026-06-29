from collections.abc import Sequence

from agents_memory.models import JsonValue

type CypherParameters = dict[str, JsonValue]


def vector_parameter(vector: Sequence[float]) -> list[JsonValue]:
    return list(vector)
