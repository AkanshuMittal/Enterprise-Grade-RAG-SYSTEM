import os
import sys
import uuid
import json
import hashlib
import logfire
from datetime import datetime, timezone
from tenacity import retry, stop_after_attempt, wait_exponential

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embeddings import embed_texts, get_embedding_dim
from app.gateway.client import portkey_client
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

logfire.configure(service_name="enterprise-ingestion-service")

PROCESSED_DATA_DIR = "processed_data"
FAILED_FILES_PATH = "failed_files.json"

CATEGORY_SAMPLE_CHARS = 2500
DEFAULT_CATEGORY = "general"

# -----------------------------------------------------------------------------
# Qdrant
# -----------------------------------------------------------------------------

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def generate_chunk_hash(text: str) -> str:
    """
    Generate a deterministic SHA-256 hash for a chunk.

    The same chunk will always generate the same hash.
    """
    normalized_text = text.strip()

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


from app.agents.nodes.planner import ALLOWED_CATEGORIES   # add this import near your other imports at the top of the file

def detect_document_category(full_text: str, filename: str) -> str:
    """
    Asks the LLM ONCE per document (not per chunk) to assign a topic
    category, constrained to ALLOWED_CATEGORIES (the same list planner.py
    uses to route queries). This keeps ingestion-time categories and
    query-time category filters in the same vocabulary — without this
    constraint, the LLM could invent a label that never matches any
    query filter, silently breaking category-filtered search.
    """
    sample = full_text[:CATEGORY_SAMPLE_CHARS]
    categories_list = "\n".join(f"- {c}" for c in ALLOWED_CATEGORIES)

    prompt = f"""
    You are labeling a document for a technical knowledge base.
    Read the excerpt below and choose the SINGLE closest matching
    category from the ALLOWED LIST below. You must pick exactly one
    value from this list — do not invent a new label, do not modify
    the spelling or wording of the chosen category.

    If nothing in the list is a reasonable match, output exactly:
    general

    ALLOWED CATEGORIES:
    {categories_list}

    DOCUMENT: {filename}
    EXCERPT:
    {sample}

    Output ONLY the chosen category string, nothing else — no
    explanation, no punctuation, no quotes.
    """

    try:
        response = portkey_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        category = response.choices[0].message.content.strip().lower()

        if category not in ALLOWED_CATEGORIES and category != "general":
            logfire.warning(
                f"LLM returned unlisted category '{category}' for "
                f"{filename} — falling back to '{DEFAULT_CATEGORY}'."
            )
            return DEFAULT_CATEGORY

        return category

    except Exception as e:
        logfire.warning(
            f"Category detection failed for {filename}: {e} — using '{DEFAULT_CATEGORY}'."
        )
        return DEFAULT_CATEGORY


def save_processed_locally(
    data: dict,
    source_type: str,
    filename: str,
) -> str:
    """
    Save parsed + chunked document metadata locally.
    """

    folder = os.path.join(
        PROCESSED_DATA_DIR,
        source_type,
    )

    os.makedirs(folder, exist_ok=True)

    dest = os.path.join(
        folder,
        f"{filename}.json",
    )

    with open(
        dest,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return dest


def save_failed_file(
    filename: str,
    file_path: str,
    source_type: str,
    error: str,
):
    """
    Save failed ingestion information so the file can be retried later.
    """

    failed_files = []

    if os.path.exists(FAILED_FILES_PATH):

        try:

            with open(
                FAILED_FILES_PATH,
                "r",
                encoding="utf-8",
            ) as f:

                failed_files = json.load(f)

        except (
            json.JSONDecodeError,
            OSError,
        ):

            failed_files = []

    # Remove previous entry for the same file
    failed_files = [
        item
        for item in failed_files
        if item.get("file_path") != file_path
    ]

    failed_files.append(
        {
            "filename": filename,
            "file_path": file_path,
            "source_type": source_type,
            "error": error,
        }
    )

    with open(
        FAILED_FILES_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            failed_files,
            f,
            ensure_ascii=False,
            indent=2,
        )

    logfire.warning(
        f"⚠️ Failed file recorded → {filename}"
    )


# -----------------------------------------------------------------------------
# Qdrant existing hash lookup
# -----------------------------------------------------------------------------

def get_existing_chunk_hashes(
    filename: str,
) -> set[str]:
    """
    Fetch existing chunk hashes for a document.

    This prevents duplicate chunks from being embedded/indexed
    when the ingestion process is run again.
    """

    # A collection that doesn't exist yet is NOT a failure — it's the
    # normal state on a fresh Qdrant cluster before anything has been
    # ingested. Treat it as "no existing chunks," same as a genuinely
    # empty result — do not retry or raise for this case.
    if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
        logfire.info(
            f"Collection '{settings.QDRANT_COLLECTION}' does not exist yet "
            f"— treating as no existing chunks for {filename}."
        )
        return set()

    return _get_existing_chunk_hashes_with_retry(filename)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _get_existing_chunk_hashes_with_retry(
    filename: str,
) -> set[str]:

    existing_hashes: set[str] = set()

    offset = None

    while True:

        records, next_offset = qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION,

            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(
                            value=filename
                        ),
                    )
                ]
            ),

            limit=100,

            offset=offset,

            with_payload=[
                "chunk_hash",
            ],

            with_vectors=False,
        )

        for record in records:

            payload = record.payload or {}

            chunk_hash = payload.get(
                "chunk_hash"
            )

            if chunk_hash:
                existing_hashes.add(
                    chunk_hash
                )

        if next_offset is None:
            break

        offset = next_offset

    return existing_hashes
    # NOTE: no broad except-and-return-empty-set here anymore.
    # A real Qdrant/network failure now retries 3x (tenacity, above)
    # and then RAISES, propagating up through process_file() and
    # main.py's ingestion runner. This is intentional: silently
    # returning an empty set on failure previously meant "treat
    # every chunk as new," which would duplicate an entire document
    # in Qdrant on every transient network blip. Failing loudly here
    # means you re-run ingestion for that one file — annoying, but
    # far cheaper than silent duplicate data polluting retrieval.


# -----------------------------------------------------------------------------
# Process single file
# -----------------------------------------------------------------------------

def process_file(
    file_path: str,
    filename: str,
    source_type: str,
):
    """
    Parse → chunk → hash → deduplicate → categorize → embed → index.
    """

    with logfire.span(
        "Processing File",
        file=filename,
        source=source_type,
    ):

        try:

            # -------------------------------------------------------------
            # 1. Extract text
            # -------------------------------------------------------------

            ext = (
                filename
                .lower()
                .rsplit(".", 1)[-1]
            )

            if ext == "pdf":

                full_text = parse_pdf(
                    file_path
                )

            elif ext in (
                "html",
                "htm",
            ):

                full_text = parse_html(
                    file_path
                )

            elif ext == "txt":

                full_text = parse_text(
                    file_path
                )

            elif ext in (
                "docx",
                "pptx",
            ):

                from app.ingestion.loaders.office import (
                    parse_office
                )

                full_text = parse_office(
                    file_path
                )

            else:

                logfire.warning(
                    f"Skipping unsupported file type: {filename}"
                )

                return

            if not full_text or not full_text.strip():

                logfire.warning(
                    f"No text extracted from {filename} — skipping."
                )

                return

            # -------------------------------------------------------------
            # 2. Chunk
            # -------------------------------------------------------------

            chunks = chunk_text(
                full_text
            )

            if not chunks:

                logfire.warning(
                    f"No chunks generated for {filename}"
                )

                return

            logfire.info(
                f"Generated {len(chunks)} chunks "
                f"for {filename}"
            )

            # -------------------------------------------------------------
            # 3. Generate hashes
            # -------------------------------------------------------------

            chunk_records = []

            for index, chunk in enumerate(chunks):

                chunk_hash = generate_chunk_hash(
                    chunk
                )

                chunk_records.append(
                    {
                        "index": index,
                        "text": chunk,
                        "chunk_hash": chunk_hash,
                    }
                )

            # -------------------------------------------------------------
            # 4. Save processed data
            # -------------------------------------------------------------

            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunk_records,
            }

            local_path = save_processed_locally(
                processed_data,
                source_type,
                filename,
            )

            logfire.info(
                f"Saved processed data → {local_path}"
            )

            # -------------------------------------------------------------
            # 5. Get existing hashes
            # -------------------------------------------------------------

            with logfire.span(
                "Checking Existing Chunks"
            ):

                existing_hashes = (
                    get_existing_chunk_hashes(
                        filename
                    )
                )

            # -------------------------------------------------------------
            # 6. Deduplicate
            # -------------------------------------------------------------

            new_chunks = [
                record
                for record in chunk_records
                if record["chunk_hash"]
                not in existing_hashes
            ]

            skipped_count = (
                len(chunk_records)
                - len(new_chunks)
            )

            logfire.info(
                f"🆕 New chunks: {len(new_chunks)} / "
                f"{len(chunk_records)}"
            )

            logfire.info(
                f"⏭️ Skipped duplicates: {skipped_count}"
            )

            # Nothing new to index
            if not new_chunks:

                logfire.info(
                    f"All chunks already exist for {filename}"
                )

                return

            # -------------------------------------------------------------
            # 6.5. Detect document category (once per document, only
            #      when there's actually new content to index — avoids
            #      wasted LLM calls on already-ingested files)
            # -------------------------------------------------------------

            with logfire.span("Detecting Category"):
                category = detect_document_category(full_text, filename)
                logfire.info(f"📁 Category for {filename}: {category}")

            # -------------------------------------------------------------
            # 6.6. Stable document_id — deterministic from filename, so
            #      re-ingesting the same file always yields the same ID
            #      (needed later for delete-by-document and versioning).
            #      NOT random uuid.uuid4() — that would create a new
            #      document_id every run, breaking any future lookup.
            # -------------------------------------------------------------

            document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, filename))
            ingested_at = datetime.now(timezone.utc).isoformat()

            # -------------------------------------------------------------
            # 7. Embed ONLY new chunks
            # -------------------------------------------------------------

            with logfire.span(
                "Vectorizing & Indexing"
            ):

                texts_to_embed = [
                    record["text"]
                    for record in new_chunks
                ]

                embeddings = embed_texts(
                    texts_to_embed
                )

                # Safety check
                if len(embeddings) != len(
                    new_chunks
                ):

                    raise RuntimeError(
                        "Embedding count does not match chunk count."
                    )

                # ---------------------------------------------------------
                # 8. Create Qdrant points
                # ---------------------------------------------------------

                points = []

                for record, vector in zip(
                    new_chunks,
                    embeddings,
                ):

                    point = models.PointStruct(

                        id=str(
                            uuid.uuid4()
                        ),

                        vector=vector,

                        payload={
                            "text": record["text"],
                            "source": filename,
                            "source_type": source_type,
                            "chunk_hash": record[
                                "chunk_hash"
                            ],
                            "chunk_index": record[
                                "index"
                            ],
                            "category": category,
                            "document_id": document_id,
                            "created_at": ingested_at,
                        },
                    )

                    points.append(point)

                # ---------------------------------------------------------
                # 9. Upsert
                # ---------------------------------------------------------

                qdrant_client.upsert(

                    collection_name=(
                        settings.QDRANT_COLLECTION
                    ),

                    points=points,
                )

                logfire.info(
                    f"✅ Indexed {len(points)} new "
                    f"points to Qdrant from {filename}."
                )

        except Exception as e:

            error_message = str(e)

            logfire.error(
                f"❌ Failed to process "
                f"{filename}: {error_message}"
            )

            save_failed_file(
                filename=filename,
                file_path=file_path,
                source_type=source_type,
                error=error_message,
            )

            # IMPORTANT:
            # Do not crash the entire ingestion job.
            return


# -----------------------------------------------------------------------------
# Process directory
# -----------------------------------------------------------------------------

def process_directory(
    dir_path: str,
    source_type: str,
):
    """
    Process every supported file in a directory.
    """

    with logfire.span(
        "Scanning Directory",
        path=dir_path,
        source=source_type,
    ):

        files = [
            f
            for f in os.listdir(dir_path)
            if os.path.isfile(
                os.path.join(
                    dir_path,
                    f,
                )
            )
        ]

        logfire.info(
            f"Found {len(files)} files in {dir_path}."
        )

        for filename in files:

            process_file(
                os.path.join(
                    dir_path,
                    filename,
                ),
                filename,
                source_type,
            )


# -----------------------------------------------------------------------------
# Main ingestion pipeline
# -----------------------------------------------------------------------------

def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: str = None,
    wipe: bool = False,
):
    """
    Scan base_dir, map sub-folders to source types,
    and ingest all documents.

    --wipe:
        Deletes the existing Qdrant collection and
        recreates it using the active embedding dimension.
    """

    with logfire.span(
        "Universal Ingestion Started",
        base_directory=base_dir,
    ):

        # -------------------------------------------------------------
        # 1. Wipe collection if requested
        # -------------------------------------------------------------

        if wipe:

            with logfire.span(
                "Wiping Collection"
            ):

                if qdrant_client.collection_exists(
                    settings.QDRANT_COLLECTION
                ):

                    qdrant_client.delete_collection(
                        settings.QDRANT_COLLECTION
                    )

                    logfire.info(
                        f"Collection "
                        f"'{settings.QDRANT_COLLECTION}' "
                        f"deleted."
                    )

        # -------------------------------------------------------------
        # 2. Create collection if missing
        # -------------------------------------------------------------

        if not qdrant_client.collection_exists(
            settings.QDRANT_COLLECTION
        ):

            dim = get_embedding_dim()

            qdrant_client.create_collection(

                collection_name=(
                    settings.QDRANT_COLLECTION
                ),

                vectors_config=models.VectorParams(

                    size=dim,

                    distance=models.Distance.COSINE,
                ),
            )

            logfire.info(
                f"Created collection "
                f"'{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )

        # -------------------------------------------------------------
        # 3. Ensure required payload indexes exist
        # -------------------------------------------------------------

        # Required for production-level filtering and deduplication:
        #   chunk_hash -> duplicate chunk detection
        #   source -> document-level filtering
        #   source_type -> true/noisy data-quality filtering
        #   category -> LLM-detected topic filtering
        #   document_id -> stable per-document filtering/deletion (Phase 5)
        # These are created automatically; no manual Qdrant UI setup is needed.

        required_indexes = {
            "chunk_hash": models.PayloadSchemaType.KEYWORD,
            "source": models.PayloadSchemaType.KEYWORD,
            "source_type": models.PayloadSchemaType.KEYWORD,
            "category": models.PayloadSchemaType.KEYWORD,
            "document_id": models.PayloadSchemaType.KEYWORD,
        }

        for field_name, field_schema in required_indexes.items():
            try:
                qdrant_client.create_payload_index(
                    collection_name=settings.QDRANT_COLLECTION,
                    field_name=field_name,
                    field_schema=field_schema,
                )

                logfire.info(
                    f"Payload index ready → {field_name} ({field_schema})."
                )

            except Exception as e:
                error_message = str(e).lower()

                # Some Qdrant/client versions may report an existing
                # index as an error. That is safe to ignore.
                if "already exists" in error_message:
                    logfire.info(
                        f"Payload index already exists → {field_name}"
                    )
                else:
                    # Do not silently continue for real schema errors.
                    # Deduplication/filtering depends on these indexes.
                    logfire.error(
                        f"Failed to create payload index '{field_name}': {e}"
                    )
                    raise

        logfire.info(
            "Qdrant collection schema verified: "
            "chunk_hash, source, source_type, category indexes ready."
        )

        # 4. Detect subdirectories
        # -------------------------------------------------------------

        subdirs = [
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(
                os.path.join(
                    base_dir,
                    d,
                )
            )
        ]

        # -------------------------------------------------------------
        # 5. No subdirectories
        # -------------------------------------------------------------

        if not subdirs:

            if explicit_source_type:

                source_type = (
                    explicit_source_type
                )

            else:

                base_name = (
                    os.path.basename(
                        os.path.normpath(
                            base_dir
                        )
                    )
                    .lower()
                )

                source_type = (
                    "true"
                    if "true" in base_name
                    else "noisy"
                    if "noisy" in base_name
                    else "general"
                )

            logfire.info(
                f"No sub-folders found — "
                f"processing '{base_dir}' "
                f"as '{source_type}'."
            )

            process_directory(
                base_dir,
                source_type,
            )

        # -------------------------------------------------------------
        # 6. Process subdirectories
        # -------------------------------------------------------------

        else:

            for subdir in subdirs:

                source_type = (
                    "true"
                    if "true" in subdir.lower()
                    else "noisy"
                    if "noisy" in subdir.lower()
                    else subdir
                )

                process_directory(

                    os.path.join(
                        base_dir,
                        subdir,
                    ),

                    source_type,
                )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Examples:
    #
    # python -m app.ingestion.processor DATA
    #
    # python -m app.ingestion.processor DATA --wipe
    #
    # python -m app.ingestion.processor DATA/true_data true

    wipe_requested = (
        "--wipe" in sys.argv
    )

    clean_args = [
        arg
        for arg in sys.argv
        if arg != "--wipe"
    ]

    target_dir = (
        clean_args[1]
        if len(clean_args) > 1
        else "DATA"
    )

    explicit_type = (
        clean_args[2]
        if len(clean_args) > 2
        else None
    )

    if not os.path.exists(
        target_dir
    ):

        print(
            f"Error: path '{target_dir}' does not exist."
        )

        sys.exit(1)

    run_universal_ingestion(

        target_dir,

        explicit_source_type=(
            explicit_type
),

        wipe=wipe_requested,
    )

    logfire.info(
        "✅ Ingestion job completed."
    )