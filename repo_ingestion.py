"""
repo_ingestion.py

Standalone reusable function to clone a GitHub repo and build a retriever from it.
Extracted from GitHub Repo Chatbot (build_vectorstore + extchecker + get_splitter),
minus Streamlit/atexit dependency so it can be dropped into any LangGraph project.
"""

import os
import shutil
import subprocess
import chromadb
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader


LANG_MAP = {
    "cpp": Language.CPP,
    "go": Language.GO,
    "java": Language.JAVA,
    "kt": Language.KOTLIN,
    "js": Language.JS,
    "ts": Language.TS,
    "php": Language.PHP,
    "proto": Language.PROTO,
    "python": Language.PYTHON,
    "py": Language.PYTHON,
    "rst": Language.RST,
    "ruby": Language.RUBY,
    "rust": Language.RUST,
    "scala": Language.SCALA,
    "swift": Language.SWIFT,
    "markdown": Language.MARKDOWN,
    "md": Language.MARKDOWN,
    "latex": Language.LATEX,
    "html": Language.HTML,
    "sol": Language.SOL,
    "csharp": Language.CSHARP,
    "cobol": Language.COBOL,
}


def _remove_repo(folder_name: str):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)


def _extchecker(file_path: Path, lang_map: dict):
    if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".pyc", ".ico"]:
        return None
    if file_path.is_file() and ".git" not in file_path.parts:
        if file_path.suffix[1:] in lang_map or file_path.suffix in [".md", ".txt"]:
            return TextLoader(file_path, autodetect_encoding=True)
    return None


def _get_splitter(ext: str, lang_map: dict):
    lang = lang_map.get(ext)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=1500, chunk_overlap=300
        )
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)


def get_repo_retriever(
    repo_url: str,
    embedding_model,
    folder_name: str = "temp_git_folder_repo",
    cleanup_after: bool = True,
):
    """
    Clone a GitHub repo, chunk its files, build an ephemeral Chroma vectorstore,
    and return (retriever, metadata_set).

    Args:
        repo_url: GitHub repo URL to clone.
        embedding_model: an already-instantiated embedding model (e.g. HuggingFaceEmbeddings).
        folder_name: local temp folder to clone into.
        cleanup_after: if True, deletes the cloned repo folder once vectorstore is built
                       (vectorstore itself is in-memory/ephemeral, so source files aren't needed after).

    Returns:
        retriever: a Chroma retriever (search_type="mmr") -> call
                    retriever.invoke(query, k=..., fetch_k=...) later.
        metadata_set: list of unique file paths that were indexed.
    """
    _remove_repo(folder_name)
    os.makedirs(folder_name, exist_ok=True)

    try:
        subprocess.run(
            ["git", "clone", repo_url, folder_name],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        _remove_repo(folder_name)
        raise ValueError(f"Invalid GitHub URL or clone failed: {e}")

    chroma_client = chromadb.EphemeralClient()

    file_chunks = []
    for file in Path(folder_name).rglob("*"):
        if file.is_file():
            loader = _extchecker(file, LANG_MAP)
            if loader is not None:
                ext = file.suffix.lstrip(".").lower()
                for doc in loader.lazy_load():
                    splitter = _get_splitter(ext, LANG_MAP)
                    file_chunks.extend(splitter.split_documents([doc]))

    if not file_chunks:
        _remove_repo(folder_name)
        raise ValueError("No indexable files found in this repo.")

    vectorstore = Chroma.from_documents(
        documents=file_chunks,
        embedding=embedding_model,
        client=chroma_client,
    )

    clean_metadata = [
        doc.get("source", "").replace(f"{folder_name}/", "")
        for doc in vectorstore.get(include=["metadatas"]).get("metadatas", [])
    ]
    metadata_set = list(set(clean_metadata))

    retriever = vectorstore.as_retriever(search_type="mmr")

    if cleanup_after:
        _remove_repo(folder_name)

    return retriever, metadata_set