"""Prompt registry: load versioned prompt templates from files (append-only, optionally SHA-identified)."""
import hashlib
import logging
from pathlib import Path
from typing import Any, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Directory containing prompt name/version folders (app/prompts/)
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def list_names() -> List[str]:
    """List available prompt names (top-level folders under prompts/)."""
    if not _PROMPTS_DIR.is_dir():
        return []
    names = []
    for p in _PROMPTS_DIR.iterdir():
        if p.is_dir() and p.name and not p.name.startswith("."):
            names.append(p.name)
    return sorted(names)


def _load_prompt_file(name: str, version: str) -> Optional[dict]:
    """Load a single prompt YAML file. Returns None if not found."""
    path = _PROMPTS_DIR / name / f"{version}.yaml"
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning(f"Failed to load prompt {name}/{version}: {e}")
        return None


def get_prompt(name: str, version: str) -> Optional[str]:
    """
    Get prompt template body by name and version.
    Returns the template string with {var} placeholders, or None if not found.
    """
    data = _load_prompt_file(name, version)
    if not data:
        return None
    body = data.get("body")
    return body.strip() if isinstance(body, str) else None


def get_prompt_with_meta(name: str, version: str) -> Optional[dict]:
    """
    Get prompt template with metadata (body, variables, description, version).
    Returns dict with keys: body, variables, description, version; or None if not found.
    """
    data = _load_prompt_file(name, version)
    if not data:
        return None
    body = data.get("body")
    if not isinstance(body, str):
        return None
    return {
        "body": body.strip(),
        "variables": data.get("variables") or [],
        "description": data.get("description") or "",
        "version": data.get("version") or version,
    }


def list_versions(name: str) -> List[str]:
    """List available version strings for a prompt name (e.g. ['v1', 'v2'])."""
    dir_path = _PROMPTS_DIR / name
    if not dir_path.is_dir():
        return []
    versions = []
    for p in dir_path.iterdir():
        if p.suffix == ".yaml" and p.stem:
            versions.append(p.stem)
    return sorted(versions)


def prompt_sha(body: str) -> str:
    """Content-derived SHA256 (hex) for a prompt body. Same content => same SHA."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def get_prompt_by_sha(name: str, content_sha: str) -> Optional[str]:
    """
    Resolve a prompt by content SHA: find first version whose body has this SHA.
    Returns template body or None.
    """
    for ver in list_versions(name):
        body = get_prompt(name, ver)
        if body and prompt_sha(body) == content_sha:
            return body
    return None


def _safe_version(version: str) -> bool:
    """Allow alphanumeric, hyphen, underscore (no path traversal)."""
    if not version or len(version) > 80:
        return False
    return all(c.isalnum() or c in "-_" for c in version)


def save_prompt(
    name: str,
    version: str,
    body: str,
    description: str = "",
    variables: Optional[List[str]] = None,
) -> bool:
    """
    Write or overwrite a prompt YAML file. Creates name dir if needed.
    Returns True on success, False on validation/IO error.
    """
    dir_path = _PROMPTS_DIR / name
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"Failed to create prompt dir {dir_path}: {e}")
        return False
    path = dir_path / f"{version}.yaml"
    data: dict[str, Any] = {
        "version": version,
        "description": description or "",
        "variables": variables if variables is not None else [],
        "body": body.strip() if body else "",
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        logger.warning(f"Failed to save prompt {name}/{version}: {e}")
        return False


def delete_prompt(name: str, version: str) -> bool:
    """Remove a prompt version file. Returns True if removed or not present."""
    path = _PROMPTS_DIR / name / f"{version}.yaml"
    if not path.is_file():
        return True
    try:
        path.unlink()
        return True
    except OSError as e:
        logger.warning(f"Failed to delete prompt {name}/{version}: {e}")
        return False


def create_prompt_name(name: str) -> bool:
    """Create a new prompt name (folder). Returns True on success."""
    dir_path = _PROMPTS_DIR / name
    if dir_path.exists():
        return True
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.warning(f"Failed to create prompt name {name}: {e}")
        return False
