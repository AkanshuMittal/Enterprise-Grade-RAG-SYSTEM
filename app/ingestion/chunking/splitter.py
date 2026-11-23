from typing import List
import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Split extracted document text into controlled-size chunks
    while trying to preserve natural text boundaries.

    Uses RecursiveCharacterTextSplitter so that very large
    paragraphs/pages cannot become a single oversized chunk.
    """

    with logfire.span("✂️ Text Chunking", text_length=len(text)):

        if not text or not text.strip():
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                " ",
                "",
            ],
        )

        chunks = splitter.split_text(text)

        valid_chunks = [
            chunk.strip()
            for chunk in chunks
            if chunk.strip()
        ]

        logfire.info(
            f"✅ Generated {len(valid_chunks)} chunks "
            f"(size={chunk_size}, overlap={chunk_overlap})"
        )

        return valid_chunks