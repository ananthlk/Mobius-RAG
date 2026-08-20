"""Regenerate the AHCA rerun status from the DATABASE, never from a tally.

Ananth, 2026-08-20: a scope document he can constantly track.

WHY IT IS GENERATED, NOT WRITTEN. A hand-maintained status doc is accurate the
moment it is written and drifting by the next stage. Worse, it drifts SILENTLY —
it keeps looking authoritative. Every number below is a live query, so the file
is either current or obviously stale by its own timestamp.

Writes two artefacts:
  docs/AHCA_RERUN_STATUS.md    human-readable, git-tracked, diffable per run
  docs/ahca_rerun_status.json  machine-readable, for Product Awareness's live view

Usage:  python3 scripts/ahca_sprint_status.py [--write]
"""
from __future__ import annotations

import json, os, sys
from datetime import datetime, timezone

import psycopg2

DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
      .replace("postgresql+asyncpg://", "postgresql://")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AHCA = ("(d.source_metadata->>'payer' ILIKE '%%AHCA%%' OR d.filename ILIKE '%%ahca%%' "
        "OR d.file_path ILIKE '%%ahca%%')")
NO_SUB = "length(regexp_replace(trim(text),'[^[:alnum:]]','','g')) < 3"


def q1(cur, sql, args=()):
    cur.execute(sql, args); r = cur.fetchone()
    return r[0] if r else 0


def collect():
    c = psycopg2.connect(DSN, connect_timeout=30); cur = c.cursor()
    cur.execute("SET statement_timeout='240s'")
    s: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    s["corpus_active"] = q1(cur, "SELECT count(*) FROM documents d WHERE d.lifecycle_state IS DISTINCT FROM 'retired'")
    s["ahca_total"] = q1(cur, f"SELECT count(*) FROM documents d WHERE d.lifecycle_state IS DISTINCT FROM 'retired' AND {AHCA}")

    # Reingested = carries captured tables OR a breadcrumb in its page text.
    s["reingested"] = q1(cur, """SELECT count(DISTINCT t.document_id) FROM document_tables t""")
    s["ahca_reingested"] = q1(cur, f"""SELECT count(DISTINCT t.document_id) FROM document_tables t
        JOIN documents d ON d.id=t.document_id WHERE {AHCA}""")
    s["tables"] = q1(cur, "SELECT count(*) FROM document_tables")
    s["table_rows"] = q1(cur, "SELECT coalesce(sum(n_rows),0) FROM document_tables")
    s["breadcrumb_pages"] = q1(cur, "SELECT count(*) FROM document_pages WHERE text LIKE '%%[Table:%%'")

    s["index_total"] = q1(cur, "SELECT count(*) FROM rag_published_embeddings")
    # Junk is an INVARIANT, and checking it is EXPENSIVE IN THE HEALTHY CASE:
    # the predicate has no index, and proving ABSENCE means scanning all 1.7M
    # rows — `LIMIT 1` does not help, because there is no first hit to stop at.
    # Both a count and an exists-probe timed out at 240s. So it is opt-in
    # (`--deep`) and reported as "not checked" otherwise, rather than quietly
    # omitted: a status doc that silently drops a health check is worse than one
    # that admits it skipped it.
    if "--deep" in sys.argv:
        cur.execute("SET statement_timeout='900s'")
        s["index_junk"] = q1(cur, f"SELECT count(*) FROM rag_published_embeddings WHERE {NO_SUB}")
        s["index_junk_checked"] = True
        cur.execute("SET statement_timeout='240s'")
    else:
        s["index_junk"] = None
        s["index_junk_checked"] = False

    # Classification freshness — the event trail, since payor_classification
    # carries no timestamp of its own.
    s["classified_today"] = q1(cur, """SELECT count(DISTINCT document_id) FROM chunking_events
        WHERE event_type='payor_classification' AND created_at > date_trunc('day', now())""")

    # Reingest as a first-class ingest source.
    cur.execute("""SELECT outcome, count(*) FROM ingest_transactions
                   WHERE source_type='reingest' GROUP BY 1""")
    s["reingest_txns"] = dict(cur.fetchall())

    cur.execute("SELECT coalesce(lifecycle_state,'unset'), count(*) FROM documents GROUP BY 1")
    s["lifecycle"] = dict(cur.fetchall())

    cur.execute("""SELECT duplicate_kind, count(*) FROM gate_decisions
        WHERE run_id = (SELECT run_id FROM gate_decisions WHERE mode='duplicates'
                        GROUP BY run_id ORDER BY max(decided_at) DESC LIMIT 1)
        GROUP BY 1 ORDER BY 2 DESC""")
    s["gate_latest"] = dict(cur.fetchall())
    s["gate_last_run_at"] = str(q1(cur, "SELECT max(decided_at) FROM gate_decisions"))

    cur.execute("""SELECT j.status, count(*) FROM chunking_jobs j
        WHERE j.created_at > now() - interval '12 hours' GROUP BY 1""")
    s["chunking_jobs_12h"] = dict(cur.fetchall())

    c.close()
    return s


def pct(a, b):
    return f"{100.0*a/b:.1f}%" if b else "—"


def render(s: dict) -> str:
    done, total = s["ahca_reingested"], s["ahca_total"]
    bar_n = int(round(20 * done / total)) if total else 0
    bar = "█" * bar_n + "░" * (20 - bar_n)
    L = []
    A = L.append
    A("# AHCA rerun — live status\n")
    A(f"**Generated** {s['generated_at']} · regenerate with "
      "`python3 scripts/ahca_sprint_status.py --write`\n")
    A("> Every number here is a live query against the database, not a tally. "
      "A hand-kept status drifts silently while still looking authoritative; this "
      "one is either current or obviously stale by its own timestamp.\n")
    A("## Progress\n")
    A(f"```\nAHCA reingested   {bar}  {done:,} / {total:,}   ({pct(done,total)})\n```\n")
    A("| | count |")
    A("|---|---|")
    A(f"| AHCA documents in scope | {total:,} |")
    A(f"| **reingested (tables captured)** | **{done:,}** |")
    A(f"| reingested corpus-wide | {s['reingested']:,} |")
    A(f"| tables captured | {s['tables']:,} |")
    A(f"| table rows captured | {s['table_rows']:,} |")
    A(f"| pages carrying a breadcrumb | {s['breadcrumb_pages']:,} |")
    A(f"| classified today | {s['classified_today']:,} |")
    A("")
    A("## Index health\n")
    A("| | count |")
    A("|---|---|")
    A(f"| published chunks | {s['index_total']:,} |")
    if s.get("index_junk_checked"):
        A(f"| **no-substance chunks** (must stay 0) | **{s['index_junk']:,}** |")
    else:
        A("| no-substance chunks | _not checked — run with `--deep`_ |")
    A(f"| corpus active | {s['corpus_active']:,} |")
    A("")
    A("## Reingest transactions\n")
    A(f"`{json.dumps(s['reingest_txns']) if s['reingest_txns'] else 'none recorded yet'}`\n")
    A("## Lifecycle\n")
    A(f"`{json.dumps(s['lifecycle'])}`\n")
    A("## Duplicate gate — latest run\n")
    A(f"last decided: `{s['gate_last_run_at']}`\n")
    A("| verdict | pairs |")
    A("|---|---|")
    for k, v in s["gate_latest"].items():
        A(f"| {k or '(none)'} | {v:,} |")
    A("")
    A("## Chunking jobs, last 12h\n")
    A(f"`{json.dumps(s['chunking_jobs_12h']) if s['chunking_jobs_12h'] else 'none'}`\n")
    A("---\n")
    A("**Scope + owners:** `docs/AHCA_RERUN_SPRINT.md` · "
      "**machine-readable:** `docs/ahca_rerun_status.json`\n")
    return "\n".join(L)


def main():
    s = collect()
    md = render(s)
    if "--write" in sys.argv:
        with open(os.path.join(ROOT, "docs", "AHCA_RERUN_STATUS.md"), "w") as f:
            f.write(md)
        with open(os.path.join(ROOT, "docs", "ahca_rerun_status.json"), "w") as f:
            json.dump(s, f, indent=2)
        print("written: docs/AHCA_RERUN_STATUS.md + docs/ahca_rerun_status.json")
    print(md)


main()
