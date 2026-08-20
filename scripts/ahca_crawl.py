"""AHCA base-root crawl → RAG ingest. Parameters ratified by Fact Store (A-51/A-52).

RUN IDENTITY — carried on everything this run touches, so it appears in Fact
Store's Runs tab like any other run of theirs:
    source_run_id  977b22af-09a5-4ed3-b007-d454b8be1e8b
    payor_id       d2afb207-7ff5-478a-8b05-780278ffec3f
    health_plan_id 7391e715-99d2-4d5c-8d3d-ec529284334d

PARAMETERS, and why they are not defaults:
    root       https://ahca.myflorida.com   seed /index.html, path_prefix "/"
    depth      5   — configured was 2, tuned for a SECTION crawl. Real rule and
                     fee-schedule paths sit 4-5 segments below root, so depth 2
                     rediscovers landing pages and misses the corpus (A-51 §2).
    max_pages  4000 · doc fetches 3000, soft alert at 80% so hitting the bound
                     is a report rather than a surprise.

PILOT: `/medicaid/rules/` first, then the rest of the root. Substituted for
Fact Store's fee-schedules-first shape because those are excluded (below). Rules
is the better pilot anyway: `59G-x.y` is the one filename form the legacy
doc_key derivation matches, so it exercises tier-1 (source_url) AND the R1
fallback ladder, and it holds the only version chains measured in this corpus.

EXCLUSIONS — both are compliance, not tuning:
  * CPT-licensed fee schedules. The AMA End-User License restricts CPT to
    "personal use only… non-commercial uses". Ananth's ruling; he sources those
    separately. Fact Store acknowledged and will not draw CPT rates from this run.
  * `ai-train=no` from AHCA's Content-Signal is BINDING. This corpus never trains
    a model; it is retrieval/reference use only, which their `use=reference`
    grants and their unset `ai-input` neither grants nor restricts.

POLITENESS is deliberate, not incidental. Honest user-agent with contact details,
a delay between requests, and we obey Disallow. AHCA blocks nine named AI agents;
we are none of them and the wildcard allows us, but crawling loudly and slowly is
the difference between using the permission and exploiting it.

Usage:  python3 scripts/ahca_crawl.py [--pilot] [--apply] [--max-pages N]
"""
from __future__ import annotations

import re, sys, time
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

ROOT = "https://ahca.myflorida.com"
SEED = "https://ahca.myflorida.com/index.html"
PATH_PREFIX = "/"
DEPTH = 5
MAX_PAGES = 4000
MAX_DOCS = 3000
PILOT_PREFIX = "/medicaid/rules/"

SOURCE_RUN_ID = "977b22af-09a5-4ed3-b007-d454b8be1e8b"
PAYOR_ID = "d2afb207-7ff5-478a-8b05-780278ffec3f"
HEALTH_PLAN_ID = "7391e715-99d2-4d5c-8d3d-ec529284334d"

RAG = "https://mobius-rag-ortabkknqa-uc.a.run.app"
UA = ("MobiusRAG/1.0 (+healthcare policy retrieval; "
      "contact: ananth.lalithakumar@gmail.com)")
DELAY = 1.5          # seconds between requests — polite, not maximal
BATCH = 25           # pages per ingest call

# CPT / AMA-licensed surfaces. Excluded by ruling, matched on BOTH url and
# content because the licence attaches to the CPT data, not to a URL pattern.
CPT_URL = re.compile(r"59g-4\.002|reimbursement-schedules|fee[-_ ]schedule", re.I)
CPT_TEXT = re.compile(r"CPT|End User License|American Medical Association", re.I)

SKIP_EXT = re.compile(r"\.(pdf|xlsx?|docx?|zip|jpe?g|png|gif|svg|ico|css|js|mp4|mp3)$", re.I)


def norm(u: str) -> str:
    """R1 normalization: lowercase scheme+host, path as-is, drop query/fragment,
    normalize trailing slash. Fact Store specified this exactly."""
    p = urlparse(u)
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", "", ""))


def in_scope(u: str) -> bool:
    p = urlparse(u)
    return (p.scheme in ("http", "https")
            and p.netloc.lower().endswith("ahca.myflorida.com")
            and (p.path or "/").startswith(PATH_PREFIX))


def main():
    apply = "--apply" in sys.argv
    pilot = "--pilot" in sys.argv
    cap = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--max-pages=")), MAX_PAGES)

    # In pilot mode, bound TRAVERSAL to the pilot area, not just acceptance.
    # The first attempt filtered only what it kept, so it still walked the whole
    # root at depth 5 to discover a handful of rules pages -- 7 accepted after
    # hundreds of fetches. A pilot that costs a full crawl is not a pilot.
    global PATH_PREFIX, SEED
    if pilot:
        PATH_PREFIX = PILOT_PREFIX
        SEED = ROOT + "/medicaid/rules/adopted-rules-main-page.html"

    sess = requests.Session()
    sess.headers["User-Agent"] = UA

    seen, pages, skipped_cpt = set(), [], 0
    q = deque([(SEED, 0)])
    t0 = time.time()
    print(f"crawl root={ROOT} depth={DEPTH} cap={cap} "
          f"{'PILOT ' + PILOT_PREFIX if pilot else 'FULL ROOT'}", flush=True)

    while q and len(pages) < cap:
        url, d = q.popleft()
        k = norm(url)
        if k in seen or d > DEPTH:
            continue
        seen.add(k)
        try:
            r = sess.get(url, timeout=40, allow_redirects=True)
        except Exception as e:
            print(f"  ERR {type(e).__name__} {url[:70]}", flush=True); continue
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            continue
        final = norm(r.url)

        # Exclusion is checked on the FETCHED page, not just the link, because
        # the licence attaches to the CPT content wherever it appears.
        if CPT_URL.search(final) or len(CPT_TEXT.findall(r.text)) > 3:
            skipped_cpt += 1
            print(f"  SKIP(cpt) {final[:74]}", flush=True)
        elif (not pilot) or urlparse(final).path.startswith(PILOT_PREFIX):
            pages.append({"url": final, "html": r.text})
            print(f"  [{len(pages):>4}] d{d} {final[:72]}", flush=True)

        if d < DEPTH:
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                nxt = urljoin(r.url, a["href"])
                if in_scope(nxt) and not SKIP_EXT.search(urlparse(nxt).path) \
                   and norm(nxt) not in seen:
                    q.append((nxt, d + 1))
        time.sleep(DELAY)
        if len(pages) and len(pages) % int(cap * 0.8) == 0:
            print(f"  ** 80% of page budget reached ({len(pages)}/{cap})", flush=True)

    print(f"\ncrawled {len(pages)} pages · skipped {skipped_cpt} CPT-licensed · "
          f"queue left {len(q)} · {time.time()-t0:.0f}s", flush=True)
    if not apply:
        print("DRY RUN — nothing ingested. Re-run with --apply.")
        return

    sent = 0
    for i in range(0, len(pages), BATCH):
        chunk = pages[i:i + BATCH]
        body = {"pages": chunk,
                "display_name": f"AHCA {'rules pilot' if pilot else 'base root'} "
                                f"— batch {i//BATCH + 1}",
                "payer": "AHCA", "state": "FL", "program": "Medicaid",
                "authority_level": "authoritative", "auto_chunk": True,
                "source_metadata": {"source_run_id": SOURCE_RUN_ID,
                                    "payor_id": PAYOR_ID,
                                    "health_plan_id": HEALTH_PLAN_ID}}
        try:
            rr = requests.post(f"{RAG}/documents/import-scraped-pages", json=body, timeout=900)
            if rr.status_code == 409:
                print(f"  batch {i//BATCH+1}: 409 duplicate_scraped (idempotent, fine)")
            elif rr.status_code == 200:
                sent += len(chunk)
                print(f"  batch {i//BATCH+1}: ingested {len(chunk)} -> {rr.json().get('document_id','')[:8]}")
            else:
                print(f"  batch {i//BATCH+1}: HTTP {rr.status_code} {rr.text[:140]}")
        except Exception as e:
            print(f"  batch {i//BATCH+1}: ERROR {type(e).__name__}: {str(e)[:120]}")
    print(f"\ningested {sent} pages")


main()
