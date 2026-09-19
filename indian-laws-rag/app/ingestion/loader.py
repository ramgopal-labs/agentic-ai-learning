import hashlib
from dataclasses import asdict, dataclass

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_CHUNK_CHARS = 2000
CHUNK_OVERLAP = 200
MIN_LAW_CHARS = 5

LEGAL_SEPARATORS = [
    "\n\n",
    "\n(1)",
    "\n(2)",
    "\n(3)",
    "\n(4)",
    "\n(5)",
    "\n(a)",
    "\n(b)",
    "\n(c)",
    "\n(d)",
    "\n(e)",
    "\n",
    ". ",
    " ",
    "",
]


@dataclass
class LawChunk:
    id: str
    act_title: str
    section: str
    parent_doc_id: str
    chunk_text: str
    parent_text: str

    def to_dict(self) -> dict:
        return asdict(self)


def make_document_id(act_title: str, section: str) -> str:
    """Create the same stable ID every time for one Act + Section."""
    value = f"{act_title}::{section}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def remove_repeated_title(act_title: str, law_text: str) -> str:
    """Remove the Act title when the dataset repeats it inside the law text."""
    if law_text.startswith(act_title):
        return law_text[len(act_title) :].lstrip()
    return law_text


def load_and_chunk_laws(limit: int | None = 200) -> list[LawChunk]:
    """
    Download Indian laws, clean each section, and split only oversized sections.

    limit=200 is useful while developing.
    Use limit=None later to index the full dataset.
    """
    dataset = load_dataset("mratanusarkar/Indian-Laws", split="train")

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        separators=LEGAL_SEPARATORS,
    )

    chunks: list[LawChunk] = []

    for row in dataset:
        act_title = (row.get("act_title") or "").strip()
        section = (row.get("section") or "").strip()
        law_text = (row.get("law") or "").strip()

        if not act_title or len(law_text) < MIN_LAW_CHARS:
            continue

        parent_text = remove_repeated_title(act_title, law_text)
        parent_doc_id = make_document_id(act_title, section)

        if len(parent_text) <= MAX_CHUNK_CHARS:
            pieces = [parent_text]
        else:
            pieces = splitter.split_text(parent_text)

        for chunk_index, piece in enumerate(pieces):
            chunks.append(
                LawChunk(
                    id=f"{parent_doc_id}_{chunk_index}",
                    act_title=act_title,
                    section=section,
                    parent_doc_id=parent_doc_id,
                    chunk_text=piece,
                    parent_text=parent_text,
                )
            )

    return chunks
