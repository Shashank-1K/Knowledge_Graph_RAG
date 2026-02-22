import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from neo4j import GraphDatabase
from config import MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASSWORD

logger = logging.getLogger(__name__)

def get_driver():
    return GraphDatabase.driver(
        MEMGRAPH_URI, auth=(MEMGRAPH_USER, MEMGRAPH_PASSWORD)
    )

def compute_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def ensure_registry_exists():
    driver = get_driver()
    try:
        with driver.session() as session:
            try:
                session.run("CREATE INDEX ON :DocRegistry(doc_id)")
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Registry setup: {e}")
    finally:
        driver.close()

def register_document(file_path: str, chunk_count: int, triple_count: int, lenses_used: List[str]) -> Dict:
    path = Path(file_path)
    file_hash = compute_file_hash(file_path)
    doc_id = file_hash[:16]
    now = datetime.now().isoformat()
    driver = get_driver()
    try:
        with driver.session() as session:
            # FIXED: Label is uniformly DocRegistry
            session.run("""
                MERGE (d:DocRegistry {doc_id: $doc_id})
                SET d.file_name = $file_name,
                    d.file_path = $file_path,
                    d.file_hash = $file_hash,
                    d.file_size_kb = $file_size_kb,
                    d.chunk_count = $chunk_count,
                    d.triple_count = $triple_count,
                    d.lenses_used = $lenses_used,
                    d.ingested_at = $ingested_at,
                    d.status = 'ingested'
            """, {
                "doc_id": doc_id,
                "file_name": path.name,
                "file_path": str(file_path),
                "file_hash": file_hash,
                "file_size_kb": round(path.stat().st_size / 1024, 1),
                "chunk_count": chunk_count,
                "triple_count": triple_count,
                "lenses_used": lenses_used,
                "ingested_at": now
            })
        logger.info(f"Registered document: {path.name} (id={doc_id})")
        return {"doc_id": doc_id, "file_name": path.name, "status": "registered"}
    except Exception as e:
        logger.error(f"Failed to register document: {e}")
        return {"error": str(e)}
    finally:
        driver.close()

def get_all_registered_docs() -> List[Dict]:
    driver = get_driver()
    docs = []
    try:
        with driver.session() as session:
            # FIXED: Label is uniformly DocRegistry
            records = session.run("""
                MATCH (d:DocRegistry)
                RETURN d.doc_id AS doc_id,
                       d.file_name AS file_name,
                       d.file_path AS file_path,
                       d.file_hash AS file_hash,
                       d.file_size_kb AS file_size_kb,
                       d.chunk_count AS chunk_count,
                       d.triple_count AS triple_count,
                       d.lenses_used AS lenses_used,
                       d.ingested_at AS ingested_at,
                       d.status AS status
                ORDER BY d.ingested_at DESC
            """)
            docs = [dict(r) for r in records]
    except Exception as e:
        logger.error(f"get_all_registered_docs: {e}")
    finally:
        driver.close()
    return docs

def is_document_ingested(file_path: str) -> Optional[Dict]:
    try:
        file_hash = compute_file_hash(file_path)
        doc_id = file_hash[:16]
    except Exception as e:
        logger.error(f"Cannot hash file {file_path}: {e}")
        return None

    driver = get_driver()
    try:
        with driver.session() as session:
            # FIXED: Syntax now properly matches the indexed doc_id property
            result = session.run("""
                MATCH (d:DocRegistry {doc_id: $doc_id})
                RETURN d.doc_id AS doc_id,
                       d.file_name AS file_name,
                       d.file_hash AS file_hash, 
                       d.ingested_at AS ingested_at,
                       d.lenses_used AS lenses_used,
                       d.chunk_count AS chunk_count,
                       d.triple_count AS triple_count,
                       d.status AS status
            """, {"doc_id": doc_id})
            
            record = result.single()
            if record:
                logger.info(f"is_document_ingested: FOUND {file_path} in graph!")
                return dict(record)
            else:
                logger.info(f"is_document_ingested: NOT FOUND {file_path}")
                return None
    except Exception as e:
        logger.error(f"is_document_ingested check error: {e}")
        return None
    finally:
        driver.close()

def get_new_documents(data_dir: str, uploaded_files: Optional[List] = None) -> Dict:
    all_files = []
    data_path = Path(data_dir)
    if data_path.exists():
        all_files.extend([
            str(f) for f in data_path.glob("*")
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in {".pdf", ".txt", ".docx", ".md", ".csv"}
        ])
    if uploaded_files:
        all_files.extend(uploaded_files)
        
    # Remove duplicates from the scan
    unique_paths = {}
    for fp in all_files:
        path_obj = Path(fp)
        unique_paths[path_obj.name] = fp
        
    all_files = list(unique_paths.values())

    new_files = []
    ingested_files = []
    changed_files = []

    for file_path in all_files:
        path = Path(file_path)
        if not path.exists():
            continue
        
        existing = is_document_ingested(file_path)
        if existing is None:
            new_files.append({"path": file_path, "name": path.name, "size_kb": round(path.stat().st_size / 1024, 1)})
        else:
            try:
                current_hash = compute_file_hash(file_path)
                if current_hash != existing.get("file_hash", ""):
                    changed_files.append({"path": file_path, "name": path.name, "size_kb": round(path.stat().st_size / 1024, 1)})
                else:
                    ingested_files.append({
                        "path": file_path,
                        "name": path.name,
                        "size_kb": round(path.stat().st_size / 1024, 1),
                        "ingested_at": existing.get("ingested_at", ""),
                        "chunk_count": existing.get("chunk_count", 0),
                        "triple_count": existing.get("triple_count", 0),
                        "lenses_used": existing.get("lenses_used", [])
                    })
            except Exception as e:
                logger.error(f"Error checking hash for {file_path}: {e}")

    return {
        "new_files": new_files,
        "ingested_files": ingested_files,
        "changed_files": changed_files,
        "total_scanned": len(all_files)
    }

def remove_document_from_registry(doc_id: str) -> bool:
    driver = get_driver()
    try:
        with driver.session() as session:
            session.run("MATCH (d:DocRegistry {doc_id: $doc_id}) DELETE d", {"doc_id": doc_id})
        return True
    except Exception as e:
        logger.error(f"remove_document_from_registry: {e}")
        return False
    finally:
        driver.close()

def get_registry_summary() -> Dict:
    docs = get_all_registered_docs()
    if not docs:
        return {"total_docs": 0, "total_chunks": 0, "total_triples": 0, "docs": []}
    return {
        "total_docs": len(docs),
        "total_chunks": sum(d.get("chunk_count", 0) for d in docs),
        "total_triples": sum(d.get("triple_count", 0) for d in docs),
        "docs": docs,
    }