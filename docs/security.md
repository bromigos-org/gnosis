# Security model

gnosis is policy-first: it exists to put an auth model, scope enforcement, and
redaction in front of a graph store that has none. This page documents the trust
boundary, token classes, and the safety properties of each surface.

## Trust boundary

Clients never touch Neo4j, Bolt, or the `neo4j-agent-memory` SDK. Every request
enters through the HTTP gateway, which authenticates it, enforces scope, applies
policy, and redacts the response. Clients receive prompt-safe `sections[]`, never
raw storage payloads. `gnosis` is the only service in the stack with graph
credentials.

## Token classes

Six token classes, each least-privilege and compared in **constant time**. Every
value comes from secret-backed deployment config (never git).

| Token | Env | Grants |
|---|---|---|
| Service | `GNOSIS_TOKEN` | normal callers: add/search/context over their own scope, and promote their own memories |
| Read operator | `GNOSIS_READ_OPERATOR_TOKEN` | operator **reads**: stats, entity/fact/preference/reasoning inspection, dedup/consolidation dry-runs |
| Write operator | `GNOSIS_WRITE_OPERATOR_TOKEN` | operator **writes**: apply dedup/consolidation, entity/fact/preference edits, buffer flush |
| Export operator | `GNOSIS_EXPORT_OPERATOR_TOKEN` | graph **export** |
| Admin operator | `GNOSIS_ADMIN_OPERATOR_TOKEN` | administrative operations |
| Federation | `GNOSIS_FEDERATION_TOKEN` | inbound federated callers only (see below); empty = inbound federation disabled |

Operator classes are separated so a read-only auditor, an apply-capable operator,
and an exporter can hold different credentials. The exact operator endpoint set is
registered in `routes/operator.py`; the caller-facing contract is in
[provider-surface.md](provider-surface.md).

## Scope enforcement

Every record carries the six-field scope spine (`tenant_id`, `space_id`,
`agent_id`, `session_id`, `user_id`, `visibility`). The gateway:

- **Isolates by `tenant_id`** — the hard boundary between deployments/businesses.
- **Keys long-term recall by `tenant_id` + `user_id`.** `agent_id` and
  `session_id` are write-side audit tags; they are stored and available in
  filtered views but do **not** partition recall.
- **Re-checks scope on every deserialized fact** during read assembly (including
  the scope-narrowed dense path), so the item budget only ever sees in-scope
  facts and a cross-scope record can never reach a prompt.

Different business entities run separate deployments with separate tenants and
storage (memory is not merged; see [federation](#federation) for consented
sharing).

## Redaction

Redaction runs before anything leaves the gateway:

- Prompt-facing content is redacted; provenance (source ids, internal scope tags)
  is kept **out of the prompt** and available only through audit read paths.
- Export, dedup, and consolidation responses are redacted.
- Reasoning traces are stored for audit and reuse, but hidden chain-of-thought is
  kept out of prompt recall and out of public/federated memory.

## Review-first operations

Destructive-looking operations are dry-run by default: dedup and consolidation
return a candidate manifest with no side effects unless explicitly applied (with a
write-operator token). Storage is append-only; supersession and dedup resolve at
read time or merge, never delete silently.

## Graph-QA safety

Natural-language graph questions are planned by an LLM but never trusted: gnosis
**validates** the generated Cypher against a scope-checked schema guide (every
`Entity`/`Fact` alias bound by `tenant_id` + `user_id`), logs it, and executes it
**read-only**. Invalid, out-of-scope, or write-attempting plans are rejected and
the read degrades to dense-only. Clients cannot submit Cypher.

## Federation safety

Federation is off by default in both directions. When enabled:

- **Consent is a required tag.** A memory crosses a boundary only if its metadata
  carries `"shareable": true`; the gateway injects that filter server-side onto
  every federated read and promote scan, regardless of caller filters.
- **Sharing is non-transitive.** Promotion strips `shareable` from the pushed
  copy, so the receiving tenant re-decides its own consent.
- **The federation token class is narrow.** An inbound `GNOSIS_FEDERATION_TOKEN`
  caller can reach only search, list, and `promoted_from`-stamped writes — every
  other route returns `403` — and cannot itself name `peers`, so federation
  cannot loop between instances.
- Promotion (push) is review-first (`dry_run=true` by default) and requires the
  normal service token, not an operator class, because callers promote their own
  scope.
