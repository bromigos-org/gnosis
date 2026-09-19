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
from gnosis.models import JsonObject

type SlotKey = tuple[str, ...]

_POINT_IN_TIME = "point_in_time"
_STATE_TEMPORAL_STATES = {"starts", "ongoing", "ends"}

# Relation-class first-word allowlist for singleton state facts.
# A "singleton" relation can have at most one true value at a time for a given
# entity (e.g. employer, home city, relationship status). Only these compete for
# a named supersession slot so that a newer "Alice works at NVIDIA" correctly
# displaces "Alice works at Google".
#
# Additive relations (likes, prefers, has_hobby, …) can have multiple concurrent
# values and must NOT be given a shared slot — collisions cause preference and
# multi-session facts to silently drop, which is worse than keeping both.
_SINGLETON_RELATION_PREFIXES: frozenset[str] = frozenset({
    # Employment / role
    "works", "employed", "employs",
    # Location / residence
    "lives", "resides", "located", "based", "moved",
    # Relationship status
    "married", "engaged", "dating", "divorced", "separated", "widowed",
    # Education (current enrolment)
    "studies", "enrolled", "graduated", "attends", "attending",
})


def is_singleton_relation_class(relation_class: str) -> bool:
    """Return True if this relation describes a singleton state.

    Uses the first word of the underscore-joined relation class as the
    discriminator ("works_at_nvidia" → "works" → singleton; "prefers_coffee"
    → "prefers" → additive).
    """
    first_word = relation_class.split("_")[0] if relation_class else ""
    return first_word in _SINGLETON_RELATION_PREFIXES


@dataclass(frozen=True, slots=True)
class FactFreshness:
    """The supersession signals for one candidate fact."""

    slot_key: SlotKey | None
    event_date: str | None
    observation_date: (
        str | None
    )  # conversation_date stored at ingest (metadata["date"])
    created_at: str | None


def slot_key(
    subject: str,
    predicate: str,
    entities: Sequence[str],
    metadata: JsonObject | None = None,
) -> SlotKey | None:
    """Compute the same-slot signature, or ``None`` when never supersedable.

    Extracted facts use relation-class slots when available (``relation_slots``
    stored at ingest for state-bearing facts): "alice:works_at" means only facts
    about Alice's employment compete, leaving location/hobby facts untouched.
    Falls back to first-entity slot for facts ingested before L-24. Returns
    ``None`` for point_in_time events — one-off events do not displace states.
    """
    normalized_subject = subject.strip().casefold()
    if not normalized_subject:
        return None
    if predicate == VERBATIM_MEMORY_PREDICATE or predicate.startswith(
        TURN_MEMORY_PREDICATE_PREFIX,
    ):
        return None
    if predicate == EXTRACTED_FACT_PREDICATE:
        if metadata is not None:
            # Prefer precise relation-class slot (stored from L-25 ingest onward)
            # but only for singleton relations — additive relations (likes, prefers,
            # has_hobby) fall through to the entity-first slot below so multiple
            # concurrent values for the same entity can coexist.
            rslots = metadata.get("relation_slots")
            if isinstance(rslots, list) and rslots:
                first = rslots[0]
                if isinstance(first, str) and first.strip():
                    relation_class = first.strip().split(":", 1)[-1]
                    if is_singleton_relation_class(relation_class):
                        return (normalized_subject, predicate, first.strip())
            # point_in_time events don't displace state facts
            ts = metadata.get("temporal_state")
            if ts == _POINT_IN_TIME:
                return None
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
    if (
        candidate.observation_date is not None
        and reference.observation_date is not None
    ):
        return candidate.observation_date > reference.observation_date
    if candidate.created_at is not None and reference.created_at is not None:
        return candidate.created_at > reference.created_at
    return False
