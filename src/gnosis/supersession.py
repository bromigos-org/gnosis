"""Deterministic read-time supersession over long-term fact candidates.

The freshness study "Don't Ask the LLM to Track Freshness" (arXiv 2606.01435)
shows a deterministic "newest wins" aggregation step beats LLM-side and
bi-temporal write-time invalidation at conflict resolution (78-94.8% vs 7%).
This module implements that step at *read* time only: when a ranked result set
carries several facts that occupy the same slot for the same user, the older
ones are dropped from the returned set and the newest is kept. Nothing is ever
mutated or deleted - storage stays append-only.

The rule is deliberately conservative: when it cannot prove two facts share a
slot, or cannot order them by recency, it keeps both. Under-superseding leaves
a stale line in the context; over-superseding silently loses information, which
is worse.

Same-slot rule:
    Two facts share a slot iff they have the same normalized ``subject`` and:
    - for extracted ``fact``-predicate memories: the same normalized first
      entity (``metadata.entities[0]``). Extracted facts all share the scope
      subject and the ``fact`` predicate, so the first entity is the cheapest
      defensible discriminator; a fact with no entity has no slot and is never
      superseded.
    - for other typed predicates: the same normalized predicate
      (``subject + predicate``, the freshness-paper slot).
    Verbatim ``memory`` and turn ``said_*`` facts are raw conversation records,
    not knowledge slots, so they never carry a slot key and are never dropped.

Recency:
    ``event_date`` metadata decides when both facts carry it, else ``created_at``.
    Both are compared as ISO-8601 strings (lexical order matches chronological
    order for a shared format). Equal or incomparable timestamps keep both.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gnosis.memory_provider import (
    EXTRACTED_FACT_PREDICATE,
    TURN_MEMORY_PREDICATE_PREFIX,
    VERBATIM_MEMORY_PREDICATE,
)

type SlotKey = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FactFreshness:
    """The supersession signals for one candidate fact."""

    slot_key: SlotKey | None
    event_date: str | None
    created_at: str | None


def slot_key(
    subject: str,
    predicate: str,
    entities: Sequence[str],
) -> SlotKey | None:
    """Compute the same-slot signature, or ``None`` when never supersedable."""
    normalized_subject = subject.strip().casefold()
    if not normalized_subject:
        return None
    if predicate == VERBATIM_MEMORY_PREDICATE or predicate.startswith(
        TURN_MEMORY_PREDICATE_PREFIX,
    ):
        return None
    if predicate == EXTRACTED_FACT_PREDICATE:
        first_entity = _first_entity(entities)
        if first_entity is None:
            return None
        return (normalized_subject, predicate, first_entity)
    normalized_predicate = predicate.strip().casefold()
    if not normalized_predicate:
        return None
    return (normalized_subject, normalized_predicate)


def drop_superseded[ItemT](
    items: Sequence[ItemT],
    freshness: Callable[[ItemT], FactFreshness],
) -> tuple[list[ItemT], int]:
    """Return items with same-slot older facts dropped, plus the drop count.

    Rank order is preserved: an item is dropped only when another item in the
    same slot is strictly newer than it. Ties and incomparable pairs keep both,
    and items with no slot key always survive.
    """
    features = [freshness(item) for item in items]
    kept: list[ItemT] = []
    dropped = 0
    for position, item in enumerate(items):
        current = features[position]
        if current.slot_key is None:
            kept.append(item)
            continue
        if any(
            other.slot_key == current.slot_key and _strictly_newer(other, current)
            for index, other in enumerate(features)
            if index != position
        ):
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


def _first_entity(entities: Sequence[str]) -> str | None:
    for entity in entities:
        normalized = entity.strip().casefold()
        if normalized:
            return normalized
    return None


def _strictly_newer(candidate: FactFreshness, reference: FactFreshness) -> bool:
    if candidate.event_date is not None and reference.event_date is not None:
        return candidate.event_date > reference.event_date
    if candidate.created_at is not None and reference.created_at is not None:
        return candidate.created_at > reference.created_at
    return False
