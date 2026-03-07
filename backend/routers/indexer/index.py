"""
routers/indexer.py

FastAPI router exposing document indexing via HTTP.

Endpoints:
    POST /api/indexer/upload — upload one or more files for indexing
    GET  /api/indexer/health — sanity check
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import read_config
from indexer import Indexer, LOADER_DISPATCH

router  = APIRouter(prefix="/api/indexer", tags=["indexer"])
indexer = Indexer(read_config())


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    namespace: str = "default",
):
    """
    Upload one or more files and index them into the vector store.

    - **files**: one or more files (PDF, TXT, CSV ....)
    - **namespace**: target collection in the vector store (default: "default")
    """
    # Validate extensions
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in LOADER_DISPATCH:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}' for '{file.filename}'. Allowed: {sorted(sorted(LOADER_DISPATCH.keys()))}"
            )

    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for file in files:
            tmp_path = Path(tmp_dir) / file.filename
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            result = indexer.ingest_file(str(tmp_path), namespace=namespace)
            results.append(result)

    return {"namespace": namespace, "files": results}