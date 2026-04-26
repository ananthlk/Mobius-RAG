"""Curator-v0 scan: build a per-domain sitemap with status + classification.

Output: /tmp/curator_sitemap.json — one entry per URL with:
  url, host, path, status_code, content_type, content_length,
  inferred_payer, inferred_authority, content_kind (page|doc),
  scrape_decision (ingest|skip|already)

Tonight this is a local script; tomorrow the same logic moves into
the curator skill behind /sources/upsert.
"""
import asyncio, json, re, sys, time
from urllib.parse import urlparse, urljoin
from collections import Counter
import httpx

UA = "Mobius-WebScraper/1.0 (+https://github.com/mobius)"
HEADERS = {"User-Agent": UA, "Accept": "application/xml,text/xml,text/html,*/*"}

# Domains we want to map tonight. Each entry can have a sub-path filter
# to keep the scan focused (AHCA's full sitemap has 1666 URLs across
# many programs; we only care about Medicaid/HCBS provider stuff).
DOMAINS = [
    {
        "host": "www.sunshinehealth.com",
        "payer": "Sunshine Health",
        "state": "FL",
        "path_prefix": None,  # crawl whole site
    },
    {
        "host": "ahca.myflorida.com",
        "payer": "AHCA",
        "state": "FL",
        # AHCA mirror is huge. Limit to medicaid coverage policies +
        # provider downloads + content/dam (asset paths).
        "path_prefix_re": re.compile(r"^/(content|medicaid|chc|hqa)", re.I),
    },
]

# Heuristics — URL path → inferred authority_level. Drives the
# `authority_level` we tag on import. Tomorrow this becomes a learned
# classifier; tonight it's regex.
AUTHORITY_PATTERNS = [
    (re.compile(r"/Billing[- _]manual", re.I),       "payer_manual"),
    (re.compile(r"/provider[- _]manual", re.I),       "payer_manual"),
    (re.compile(r"/utilization[- _]management", re.I),"payer_policy"),
    (re.compile(r"/payment[- _]polic", re.I),         "payer_policy"),
    (re.compile(r"/clinical[- _]polic", re.I),        "payer_policy"),
    (re.compile(r"/criteria/", re.I),                 "payer_policy"),
    (re.compile(r"/preauth", re.I),                   "payer_policy"),
    (re.compile(r"/member", re.I),                    "member_handbook"),
    (re.compile(r"\.pdf$", re.I),                     None),  # leave as-is
]

DOC_EXTENSIONS = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx")


async def fetch(client, url, *, method="GET"):
    try:
        r = await client.request(method, url, follow_redirects=True, timeout=20)
        return r.status_code, dict(r.headers), (r.text if method == "GET" else "")
    except httpx.RequestError as e:
        return -1, {"_error": type(e).__name__}, ""


async def fetch_sitemap(client, origin):
    """Fetch and parse sitemap.xml or sitemap_index.xml. Returns list of URLs."""
    from bs4 import BeautifulSoup
    for path in ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml"):
        sc, hdr, body = await fetch(client, origin.rstrip("/") + path)
        if sc != 200 or not body:
            continue
        soup = BeautifulSoup(body, features="xml")
        locs = [(loc.text or "").strip() for loc in soup.find_all("loc")]
        locs = [u for u in locs if u.startswith(("http://", "https://"))]
        if not locs:
            continue
        # Sitemap-index? recurse one level
        if locs[0].lower().rstrip("/").endswith(".xml"):
            children = []
            for child_sm in locs[:50]:
                csc, _, cbody = await fetch(client, child_sm)
                if csc != 200:
                    continue
                csoup = BeautifulSoup(cbody, features="xml")
                children.extend(
                    (l.text or "").strip()
                    for l in csoup.find_all("loc")
                    if (l.text or "").strip().startswith(("http://", "https://"))
                )
            return children
        return locs
    return []


def classify(url):
    p = urlparse(url)
    is_doc = p.path.lower().endswith(DOC_EXTENSIONS)
    auth = None
    for rx, lvl in AUTHORITY_PATTERNS:
        if rx.search(p.path) and lvl:
            auth = lvl
            break
    return {
        "content_kind": "doc" if is_doc else "page",
        "inferred_authority_level": auth,
        "extension": p.path.rsplit(".", 1)[-1].lower() if "." in p.path else "",
    }


async def probe_url(client, url):
    """HEAD first (cheap). If HEAD 405/disallowed, fall back to GET."""
    sc, hdr, _ = await fetch(client, url, method="HEAD")
    if sc in (405, 501) or sc == -1:
        sc, hdr, _ = await fetch(client, url, method="GET")
    return {
        "status_code": sc,
        "content_type": (hdr.get("content-type") or "").split(";")[0].strip(),
        "content_length": int(hdr.get("content-length") or 0) if hdr.get("content-length", "").isdigit() else 0,
    }


async def scan_domain(domain_cfg):
    print(f"\n=== Scanning {domain_cfg['host']} ===", file=sys.stderr)
    origin = f"https://{domain_cfg['host']}"
    async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
        # 1. Pull sitemap
        urls = await fetch_sitemap(client, origin)
        print(f"  sitemap yielded {len(urls)} URLs", file=sys.stderr)
        # 2. Filter by path_prefix
        if "path_prefix_re" in domain_cfg:
            rx = domain_cfg["path_prefix_re"]
            urls = [u for u in urls if rx.search(urlparse(u).path)]
            print(f"  filtered to {len(urls)} matching {rx.pattern!r}", file=sys.stderr)
        elif domain_cfg.get("path_prefix"):
            urls = [u for u in urls if urlparse(u).path.startswith(domain_cfg["path_prefix"])]
            print(f"  filtered to {len(urls)} under {domain_cfg['path_prefix']!r}", file=sys.stderr)
        # 3. Probe each (concurrent, bounded)
        sem = asyncio.Semaphore(20)
        async def _one(u):
            async with sem:
                meta = await probe_url(client, u)
                return {
                    "url": u,
                    "host": domain_cfg["host"],
                    "payer": domain_cfg["payer"],
                    "state": domain_cfg["state"],
                    **classify(u),
                    **meta,
                }
        t0 = time.monotonic()
        results = await asyncio.gather(*[_one(u) for u in urls])
        print(f"  probed {len(results)} URLs in {time.monotonic()-t0:.1f}s", file=sys.stderr)
        return results


async def main():
    all_results = []
    for cfg in DOMAINS:
        all_results.extend(await scan_domain(cfg))
    json.dump(all_results, open("/tmp/curator_sitemap.json", "w"), indent=2)
    # ── Summary ──
    print("\n=== SUMMARY ===")
    print(f"total URLs scanned: {len(all_results)}")
    by_host = Counter(r["host"] for r in all_results)
    print(f"by host: {dict(by_host)}")
    by_kind = Counter(r["content_kind"] for r in all_results)
    print(f"by kind: {dict(by_kind)}")
    by_status = Counter(r["status_code"] for r in all_results)
    print(f"by status: {dict(by_status)}")
    blocked = [r for r in all_results if r["status_code"] in (401, 403, 451)]
    not_found = [r for r in all_results if r["status_code"] in (404, 410)]
    print(f"blocked (401/403/451): {len(blocked)}")
    print(f"missing (404/410):     {len(not_found)}")
    docs_ok = [r for r in all_results if r["content_kind"] == "doc" and r["status_code"] == 200]
    pages_ok = [r for r in all_results if r["content_kind"] == "page" and r["status_code"] == 200]
    print(f"reachable docs:  {len(docs_ok)}")
    print(f"reachable pages: {len(pages_ok)}")
    # By authority hint
    by_auth = Counter(r.get("inferred_authority_level") or "(none)" for r in all_results)
    print(f"\ninferred authority distribution: {dict(by_auth)}")

asyncio.run(main())
