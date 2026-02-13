"""
Policy lexicon repository: load/update approved tags and export lexicon.

Uses in-DB tables policy_lexicon_meta and policy_lexicon_entries (see migrations).
"""
from __future__ import annotations

from types import SimpleNamespace
from sqlalchemy import text


async def load_lexicon_snapshot_db(db):
    """Load current p/d/j lexicon from DB. Returns object with .version, .meta, .p_tags, .d_tags, .j_tags."""
    meta_row = await db.execute(
        text(
            "SELECT revision, lexicon_version, lexicon_meta FROM policy_lexicon_meta ORDER BY updated_at DESC NULLS LAST LIMIT 1"
        )
    )
    meta = meta_row.fetchone()
    if not meta:
        return SimpleNamespace(
            version="v1",
            meta={"revision": 0, "lexicon_version": "v1"},
            p_tags={},
            d_tags={},
            j_tags={},
        )
    revision = int(meta[0] or 0)
    lexicon_version = str(meta[1] or "v1")
    lexicon_meta = meta[2] if isinstance(meta[2], dict) else {}
    if not isinstance(lexicon_meta, dict):
        lexicon_meta = {}
    meta_dict = {"revision": revision, "lexicon_version": lexicon_version, **lexicon_meta}

    entries_row = await db.execute(
        text(
            "SELECT kind, code, parent_code, spec FROM policy_lexicon_entries WHERE active = true ORDER BY kind, code"
        )
    )
    entries = entries_row.fetchall()

    def build_nested(entries_list):
        by_kind = {"p": {}, "d": {}, "j": {}}
        for e in entries_list:
            kind, code, parent_code, spec = e[0], e[1], e[2], e[3]
            if kind not in by_kind:
                continue
            spec = spec if isinstance(spec, dict) else {}
            by_kind[kind][str(code)] = dict(spec)
        # Add children structure if parent_code used
        for kind in ("p", "d", "j"):
            root = {}
            for code, spec in by_kind[kind].items():
                root[code] = dict(spec)
                root[code].setdefault("children", {})
            by_kind[kind] = root
        return by_kind["p"], by_kind["d"], by_kind["j"]

    p_tags, d_tags, j_tags = build_nested(entries)
    return SimpleNamespace(
        version=lexicon_version,
        meta=meta_dict,
        p_tags=p_tags,
        d_tags=d_tags,
        j_tags=j_tags,
    )


async def approve_phrase_to_db(db, *, kind: str, normalized: str, target_code: str | None = None, tag_spec: dict | None = None):
    """Approve a candidate phrase into the lexicon. Inserts or updates policy_lexicon_entries."""
    kind = (kind or "d").strip().lower()
    if kind not in ("p", "d", "j"):
        kind = "d"
    code = (target_code or normalized[:500] or "unknown").strip() or "unknown"
    code = code.replace(" ", "_").lower()[:500]
    spec = dict(tag_spec) if isinstance(tag_spec, dict) else {}
    if "phrases" not in spec:
        spec["phrases"] = [normalized.strip()] if normalized else []
    if "description" not in spec:
        spec["description"] = normalized.strip()[:500] if normalized else ""

    await db.execute(
        text(
            """
            INSERT INTO policy_lexicon_entries (kind, code, spec, active)
            VALUES (:kind, :code, CAST(:spec AS jsonb), true)
            ON CONFLICT (kind, code) DO UPDATE SET
                spec = CAST(:spec AS jsonb),
                updated_at = (NOW() AT TIME ZONE 'utc')
            """
        ),
        {"kind": kind, "code": code, "spec": __jsonb(spec)},
    )
    await db.flush()
    return {"kind": kind, "code": code, "action": "approved"}


def __jsonb(d):
    import json
    return json.dumps(d)


async def bump_revision(db):
    """Bump lexicon revision in policy_lexicon_meta."""
    await db.execute(
        text(
            "UPDATE policy_lexicon_meta SET revision = COALESCE(revision, 0) + 1, updated_at = (NOW() AT TIME ZONE 'utc')"
        )
    )
    await db.flush()
    r = await db.execute(text("SELECT revision FROM policy_lexicon_meta ORDER BY updated_at DESC LIMIT 1"))
    row = r.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


async def export_yaml_from_db(db):
    """Export lexicon to YAML file (optional). Returns path or None."""
    try:
        import yaml
        from pathlib import Path
        lex = await load_lexicon_snapshot_db(db)
        out = {
            "version": lex.version,
            "meta": lex.meta,
            "p_tags": lex.p_tags,
            "d_tags": lex.d_tags,
            "j_tags": lex.j_tags,
        }
        path = Path("data") / "policy_lexicon.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(out, f, default_flow_style=False, allow_unicode=True)
        return str(path)
    except Exception:
        return None


async def update_tag_in_db(db, *, kind: str, code: str, spec: dict | None = None, active: bool | None = None):
    """Update a tag's spec or active state in policy_lexicon_entries."""
    kind = (kind or "").strip().lower()
    code = (code or "").strip()
    if not kind or not code:
        raise KeyError("kind and code required")
    updates = []
    params = {"kind": kind, "code": code}
    if spec is not None:
        import json
        updates.append("spec = CAST(:spec AS jsonb)")
        params["spec"] = json.dumps(spec) if isinstance(spec, dict) else "{}"
    if active is not None:
        updates.append("active = :active")
        params["active"] = active
    if not updates:
        return
    await db.execute(
        text(
            f"UPDATE policy_lexicon_entries SET {', '.join(updates)}, updated_at = (NOW() AT TIME ZONE 'utc') WHERE kind = :kind AND code = :code"
        ),
        params,
    )
    await db.flush()
