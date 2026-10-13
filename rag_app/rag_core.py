import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = str(DATA_DIR / "chroma_db")
MANIFEST_PATH = DATA_DIR / "manifest.json"

os.environ.setdefault("USER_AGENT", "generative-ai-projects-rag-app/1.0")

SUPPORTED_FILE_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Source:
    id: str
    kind: str  # "file" | "link"
    name: str  # display name: filename or URL
    location: str  # file path on disk, or the URL
    status: str = "pending"  # "pending" | "synced" | "error"
    error: str | None = None
    added_at: str = ""
    synced_at: str | None = None


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest() -> list[Source]:
    _ensure_dirs()
    if not MANIFEST_PATH.exists():
        return []
    raw = json.loads(MANIFEST_PATH.read_text())
    return [Source(**item) for item in raw]


def save_manifest(sources: list[Source]) -> None:
    _ensure_dirs()
    MANIFEST_PATH.write_text(json.dumps([asdict(s) for s in sources], indent=2))


def add_file_source(filename: str, file_bytes: bytes) -> Source:
    _ensure_dirs()
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FILE_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix} (allowed: {', '.join(sorted(SUPPORTED_FILE_SUFFIXES))})")

    source_id = uuid.uuid4().hex
    dest = UPLOADS_DIR / f"{source_id}_{filename}"
    dest.write_bytes(file_bytes)

    source = Source(
        id=source_id,
        kind="file",
        name=filename,
        location=str(dest),
        added_at=_now(),
    )
    sources = load_manifest()
    sources.append(source)
    save_manifest(sources)
    return source


def add_link_source(url: str) -> Source:
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("Link must start with http:// or https://")

    source = Source(
        id=uuid.uuid4().hex,
        kind="link",
        name=url,
        location=url,
        added_at=_now(),
    )
    sources = load_manifest()
    sources.append(source)
    save_manifest(sources)
    return source


def remove_source(source_id: str) -> None:
    sources = load_manifest()
    remaining = []
    for s in sources:
        if s.id == source_id:
            if s.kind == "file" and Path(s.location).exists():
                Path(s.location).unlink()
        else:
            remaining.append(s)
    save_manifest(remaining)


def _load_documents(source: Source):
    if source.kind == "link":
        loader = WebBaseLoader(source.location)
    else:
        suffix = Path(source.location).suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(source.location)
        elif suffix == ".docx":
            loader = Docx2txtLoader(source.location)
        else:
            loader = TextLoader(source.location)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_id"] = source.id
        doc.metadata["source_name"] = source.name
    return docs


_embeddings = None


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return _embeddings


def get_vectorstore() -> Chroma:
    _ensure_dirs()
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=_get_embeddings())


def has_synced_data() -> bool:
    return any(s.status == "synced" for s in load_manifest())


def sync_pending(progress_callback=None) -> list[Source]:
    """Embed every pending source. Returns the updated list of sources touched."""
    sources = load_manifest()
    pending = [s for s in sources if s.status != "synced"]
    if not pending:
        return []

    vectorstore = get_vectorstore()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    touched = []
    for i, source in enumerate(pending):
        if progress_callback:
            progress_callback(i, len(pending), source)
        try:
            docs = _load_documents(source)
            chunks = splitter.split_documents(docs)
            vectorstore.add_documents(chunks)
            source.status = "synced"
            source.error = None
            source.synced_at = _now()
        except Exception as exc:  # noqa: BLE001 - surface any loader/embedding failure to the UI
            source.status = "error"
            source.error = str(exc)
        touched.append(source)

    all_sources = load_manifest()
    by_id = {s.id: s for s in all_sources}
    for t in touched:
        by_id[t.id] = t
    save_manifest(list(by_id.values()))
    return touched


_llm = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    return _llm


_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the answer is not contained in the context, reply exactly: "
    "\"Not given in the uploaded data.\"\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def _extract_text(response) -> str:
    if isinstance(response.content, str):
        return response.content
    return "".join(
        block["text"] for block in response.content if isinstance(block, dict) and block.get("type") == "text"
    )


def ask(question: str, k: int = 4) -> tuple[str, list]:
    if not has_synced_data():
        return "Not given in the uploaded data.", []

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    if not docs:
        return "Not given in the uploaded data.", []

    context = "\n\n".join(d.page_content for d in docs)
    messages = _PROMPT.invoke({"context": context, "question": question})
    response = _get_llm().invoke(messages)
    return _extract_text(response), docs
