from collections.abc import Sequence

from gnosis.models import JsonValue

type CypherParameters = dict[str, JsonValue]


def vector_parameter(vector: Sequence[float]) -> list[JsonValue]:
    return list(vector)
