"""Write-path policy gates: add modes, extraction flags, preview sources.

Pure request/settings validation for the ingestion surfaces. These helpers
decide whether an add is verbatim or inferred, which extraction features a
write may use (entity extraction implies-relations rule plus the service
flags), which model each LLM collaborator resolves to, and whether preview
sources (raw text, OCR images, RustFS references) are allowed by service
policy. Violations raise :class:`~gnosis.backend_protocols.BackendRequestError`
with an operator-safe detail.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from gnosis.backend_protocols import BackendRequestError
from gnosis.models import (
    ExtractionCandidate,
    ExtractionPreviewRequest,
    JsonObject,
    MemoryAddRequest,
    MessageWriteRequest,
    RustFSSourceReference,
)
from gnosis.settings import Settings

_PREVIEW_WRITE_DETAIL: Final[str] = (
    "Use /v1/memory/extraction/preview for dry-run previews."
)
_RELATION_ENTITY_DETAIL: Final[str] = "Relation extraction requires entity extraction."
_RAW_TEXT_PREVIEW_DETAIL: Final[str] = (
    "Raw text extraction is available only through preview."
)
_OCR_PREVIEW_ONLY_DETAIL: Final[str] = (
    "OCR extraction is available only through preview."
)
_PREVIEW_DISABLED_DETAIL: Final[str] = (
    "Extraction preview is disabled by service policy."
)
_OCR_DISABLED_DETAIL: Final[str] = "OCR preview is disabled by service policy."
_OCR_SIZE_DETAIL: Final[str] = "OCR image exceeds service policy size limit."
_RUSTFS_DISABLED_DETAIL: Final[str] = (
    "RustFS source references are disabled by service policy."
)
_RUSTFS_BUCKET_DETAIL: Final[str] = "RustFS source bucket is outside service policy."
_RUSTFS_KEY_DETAIL: Final[str] = "RustFS source key is outside service policy."
_MEMORY_MODE_DETAIL: Final[str] = (
    "Provide messages with infer=true or content with infer=false."
)
_MEMORY_MESSAGES_INFER_DETAIL: Final[str] = "messages require infer=true."
_MEMORY_CONTENT_INFER_DETAIL: Final[str] = "content requires infer=false."


@dataclass(frozen=True, slots=True)
class ExtractionPolicy:
    extract_entities: bool
    extract_relations: bool


def require_memory_add_mode(request: MemoryAddRequest) -> None:
    has_messages = bool(request.messages)
    has_content = request.content is not None
    if has_messages == has_content:
        raise BackendRequestError(_MEMORY_MODE_DETAIL)
    if has_messages and not request.infer:
        raise BackendRequestError(_MEMORY_MESSAGES_INFER_DETAIL)
    if has_content and request.infer:
        raise BackendRequestError(_MEMORY_CONTENT_INFER_DETAIL)


def message_extraction_policy(
    request: MessageWriteRequest,
    settings: Settings,
) -> ExtractionPolicy:
    if request.preview_extraction:
        raise BackendRequestError(_PREVIEW_WRITE_DETAIL)
    return extraction_policy(
        extract_entities=request.extract_entities,
        extract_relations=request.extract_relations,
        settings=settings,
    )


def preview_extraction_policy(
    request: ExtractionPreviewRequest,
    settings: Settings,
) -> ExtractionPolicy:
    return extraction_policy(
        extract_entities=request.extract_entities,
        extract_relations=request.extract_relations,
        settings=settings,
    )


def extraction_policy(
    *,
    extract_entities: bool | None,
    extract_relations: bool | None,
    settings: Settings,
) -> ExtractionPolicy:
    if extract_relations is True and extract_entities is not True:
        raise BackendRequestError(_RELATION_ENTITY_DETAIL)
    entities_enabled = (
        extract_entities is True and settings.gnosis_extract_entities_enabled
    )
    relations_enabled = (
        extract_relations is True
        and entities_enabled
        and settings.gnosis_extract_relations_enabled
    )
    return ExtractionPolicy(
        extract_entities=entities_enabled,
        extract_relations=relations_enabled,
    )


def fact_extraction_model(settings: Settings) -> str:
    return settings.gnosis_fact_extraction_model or settings.gnosis_llm


def sufficiency_model(settings: Settings) -> str:
    return settings.gnosis_sufficiency_model or settings.gnosis_llm


def routing_model(settings: Settings) -> str:
    return settings.gnosis_routing_model or settings.gnosis_llm


def conversation_date(caller_metadata: JsonObject) -> str:
    """Resolve the prompt's conversation date for relative-date resolution.

    Callers that replay historical sessions (membench) supply
    ``metadata.session_date``; live ingestion falls back to the ingest date.
    """
    session_date = caller_metadata.get("session_date")
    if isinstance(session_date, str) and session_date:
        return session_date
    return datetime.now(UTC).date().isoformat()


def require_ingestion_sources_allowed(
    request: MessageWriteRequest,
    settings: Settings,
) -> None:
    if request.raw_text_documents:
        raise BackendRequestError(_RAW_TEXT_PREVIEW_DETAIL)
    if request.ocr_image_references:
        raise BackendRequestError(_OCR_PREVIEW_ONLY_DETAIL)
    _require_rustfs_references_allowed(request.rustfs_source_references, settings)


def require_preview_enabled(settings: Settings) -> None:
    if not settings.gnosis_extraction_preview_enabled:
        raise BackendRequestError(_PREVIEW_DISABLED_DETAIL)


def require_preview_sources_allowed(
    request: ExtractionPreviewRequest,
    settings: Settings,
) -> None:
    if request.ocr_image_references and not settings.gnosis_ocr_enabled:
        raise BackendRequestError(_OCR_DISABLED_DETAIL)
    for image in request.ocr_image_references:
        if image.size_bytes > settings.gnosis_ocr_max_image_bytes:
            raise BackendRequestError(_OCR_SIZE_DETAIL)
        if image.rustfs is not None:
            _require_rustfs_references_allowed([image.rustfs], settings)
    _require_rustfs_references_allowed(request.rustfs_source_references, settings)


def _require_rustfs_references_allowed(
    references: list[RustFSSourceReference],
    settings: Settings,
) -> None:
    if not references:
        return
    if not settings.gnosis_rustfs_enabled:
        raise BackendRequestError(_RUSTFS_DISABLED_DETAIL)
    for reference in references:
        if (
            settings.gnosis_rustfs_bucket
            and reference.bucket != settings.gnosis_rustfs_bucket
        ):
            raise BackendRequestError(_RUSTFS_BUCKET_DETAIL)
        if settings.gnosis_rustfs_prefix and not reference.object_key.startswith(
            settings.gnosis_rustfs_prefix,
        ):
            raise BackendRequestError(_RUSTFS_KEY_DETAIL)


def preview_document_count(request: ExtractionPreviewRequest) -> int:
    count = len(request.raw_text_documents)
    if request.content is not None:
        count += 1
    return count


def preview_source_ids(request: ExtractionPreviewRequest) -> list[str]:
    source_ids: list[str] = []
    if request.content is not None:
        source_ids.append("message.content")
    source_ids.extend(document.source_id for document in request.raw_text_documents)
    source_ids.extend(image.source_id for image in request.ocr_image_references)
    source_ids.extend(
        f"rustfs://{reference.bucket}/{reference.object_key}"
        for reference in request.rustfs_source_references
    )
    return source_ids


def preview_candidates(
    request: ExtractionPreviewRequest,
    settings: Settings,
) -> list[ExtractionCandidate]:
    candidates: list[ExtractionCandidate] = []
    if request.content is not None:
        candidates.append(
            ExtractionCandidate(
                kind="text_chunk",
                text=_preview_text(request.content),
                source_id="message.content",
                confidence=1.0,
            ),
        )
    candidates.extend(
        ExtractionCandidate(
            kind="text_chunk",
            text=_preview_text(document.text),
            source_id=document.source_id,
            confidence=1.0,
        )
        for document in request.raw_text_documents
    )
    if settings.gnosis_ocr_enabled:
        candidates.extend(
            ExtractionCandidate(
                kind="ocr_text",
                text=f"OCR preview placeholder via {settings.gnosis_ocr_model}",
                source_id=image.source_id,
                confidence=0.0,
            )
            for image in request.ocr_image_references
        )
    return candidates


def _preview_text(text: str) -> str:
    return text[:240]
