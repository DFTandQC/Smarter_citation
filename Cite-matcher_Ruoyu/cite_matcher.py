# cite_matcher.py auto-matches citations for a given paragraph using Kimi, WOS, Crossref, and OpenAlex. 
import os, re, json, time, math, yaml, pathlib, requests, argparse, textwrap, unicodedata, sys
from datetime import datetime, timezone
from typing import List, Dict, Any
from kimi_extract_query import KimiExtractor

# ============ 读取配置 ============
CFG = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))

LLM_BASE_URL = CFG["llm"]["base_url"]
LLM_API_KEY  = CFG["llm"]["api_key"]
LLM_MODEL    = CFG["llm"]["model"]

WOS_URL      = CFG["wos"]["api_url"]
WOS_KEY      = CFG["wos"]["api_key"]
YEARS_BACK   = int(CFG["wos"]["years_back"])
WL_JOURNALS  = [j.lower() for j in CFG["wos"]["journal_whitelist"]]
WOS_DB       = CFG["wos"].get("database_id", "WOS")
WOS_LIMIT    = int(CFG["wos"].get("limit", 50))

CR_URL       = CFG["crossref"]["works_endpoint"]
CR_MAILTO    = CFG["crossref"]["polite_mailto"]
CR_ROWS      = int(CFG["crossref"].get("rows", 50))
OALEX_URL    = CFG["openalex"]["works_endpoint"]
OA_PER_PAGE  = int(CFG["openalex"].get("per_page", 50))

UNPAY_URL    = CFG["unpaywall"]["endpoint"]
UNPAY_EMAIL  = CFG["unpaywall"]["email"]

OUT_DIR      = pathlib.Path(CFG["output"]["out_dir"]); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL    = pathlib.Path(CFG["output"]["results_json"])
OUT_BIB      = pathlib.Path(CFG["output"]["bibtex_path"])
OUT_RIS      = pathlib.Path(CFG["output"]["ris_path"])
TOP_N        = int(CFG["output"]["top_n"])
KIMI_CACHE_PATH = OUT_DIR / "kimi_cache.json"

# Controls for Kimi/LLM usage. Can be overridden by CLI args later.
MAX_KIMI_MULTIPLIER = 2
KIMI_ENABLED = True

NOW_YEAR     = datetime.now(timezone.utc).year


def sanitize_keyword(k: str) -> str:
    """Normalize a keyword for use in WOS queries.

    - Normalize unicode to NFKC
    - Replace en/em dashes with ASCII hyphen
    - Remove non-ASCII characters (encoded away) to avoid API rejecting them
    - Collapse whitespace
    """
    if not k:
        return k
    k = unicodedata.normalize("NFKC", k)
    # replace common dash characters with ASCII hyphen
    k = k.replace('\u2013', '-').replace('\u2014', '-')
    # remove control chars and normalize spaces
    k = re.sub(r'[\r\n\t]+', ' ', k)
    # drop non-ascii to avoid unexpected percent-encodings the API might reject
    k = k.encode('ascii', 'ignore').decode('ascii')
    k = re.sub(' +', ' ', k).strip()
    return k

# ============ 小工具 ============
def norm(s: str) -> str:
    return (s or "").strip()

def safe_get(d: dict, *keys, default=None):
    for k in keys:
        if d is None: return default
        d = d.get(k)
    return d if d is not None else default

def dedup_by_doi(items: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for it in items:
        doi = (it.get("doi") or "").lower()
        if not doi:
            continue
        if doi in seen:
            continue
        seen.add(doi); out.append(it)
    return out

# ============ 0) Kimi 抽取检索意图 ============
def extract_intents(paragraph: str) -> dict:
    extractor = KimiExtractor(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, model=LLM_MODEL)
    intents = extractor.extract(paragraph, years_back=YEARS_BACK)
    # 兜底：若 LLM 未给时间窗
    if "year_from" not in intents or "year_to" not in intents:
        intents["year_from"] = NOW_YEAR - YEARS_BACK
        intents["year_to"] = NOW_YEAR
    # 清洗关键词长度
    intents["keywords"] = [k for k in (intents.get("keywords") or []) if len(k.strip())>1][:15]
    intents["topics"]   = intents.get("topics") or []
    return intents

# ============ 1) WOS 检索 ============
def wos_search(intents: Dict[str,Any]) -> List[Dict]:
    assert WOS_KEY, "Missing WOS API key in config.yaml"
    kws = intents.get("keywords") or intents.get("topics") or []
    if not kws:
        return []
    # sanitize keywords to avoid unicode characters (e.g. en-dash) that may cause API 400
    kws = [sanitize_keyword(k) for k in kws]
    kws = [k for k in kws if k]
    if not kws:
        return []
    # build query: prefer TS=(term) but if API rejects (400) we'll retry quoting multi-word terms
    def build_q(use_quotes: bool = False) -> str:
        parts = []
        for k in kws:
            if use_quotes and ' ' in k:
                parts.append(f'TS=("{k}")')
            else:
                parts.append(f"TS=({k})")
        return " OR ".join(parts)

    q = build_q(use_quotes=False)
    year_from = intents.get("year_from")
    year_to   = intents.get("year_to")

    params = {
        "databaseId": WOS_DB,
        "usrQuery": q,
        "from": 0,
        "limit": WOS_LIMIT,
        "sort": "load_date:desc",
        "publication_year_from": year_from,
        "publication_year_to": year_to
    }
    headers = {"X-ApiKey": WOS_KEY, "Accept": "application/json", "Content-Type": "application/json"}
    out = []
    tried_quoted = False
    while True:
        try:
            # WOS API expects query in JSON body, not URL params
            r = requests.post(WOS_URL, json=params, headers=headers, timeout=60)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # if WOS returns 400, retry once with quoted multi-word terms
            status = getattr(e.response, 'status_code', None)
            body = None
            try:
                body = e.response.text
            except Exception:
                body = None
            print(f"WOS HTTP error {status}. Response body: {body}", file=sys.stderr)
            if status == 400 and not tried_quoted:
                tried_quoted = True
                q = build_q(use_quotes=True)
                params["usrQuery"] = q
                print("Retrying WOS query with quoted multi-word keywords...", file=sys.stderr)
                time.sleep(0.2)
                continue
            raise
        data = r.json()
        hits = data.get("hits") or data.get("data") or []
        for it in hits:
            src = it.get("source") or {}
            jtitle = src.get("title") or it.get("sourceTitle") or ""
            out.append({
                "title": it.get("title") or it.get("docTitle"),
                "doi": it.get("doi"),
                "journal": jtitle,
                "year": safe_get(it, "publication_year") or safe_get(it, "publication_date") or "",
                "authors": [a.get("full_name") or a.get("name") for a in (it.get("authors") or []) if a],
                "source": "wos"
            })
        nxt = data.get("next")
        if nxt:
            params["from"] = nxt
            time.sleep(0.25)
        else:
            break
    return out

# ============ 2) Crossref / OpenAlex ============
def crossref_search(intents: Dict[str,Any]) -> List[Dict]:
    kw = " ".join((intents.get("keywords") or [])[:8]) or " ".join(intents.get("topics") or [])
    if not kw: return []
    year_from = intents.get("year_from")
    year_to   = intents.get("year_to")
    params = {"query": kw, "filter": f"from-pub-date:{year_from}-01-01,until-pub-date:{year_to}-12-31", "rows": CR_ROWS}
    headers = {"User-Agent": f"cite-matcher (mailto:{CR_MAILTO})"}
    r = requests.get(CR_URL, params=params, headers=headers, timeout=60)
    if r.status_code != 200: return []
    items = safe_get(r.json(), "message", "items", default=[]) or []
    out = []
    for it in items:
        out.append({
            "title": (it.get("title") or [""])[0],
            "doi": it.get("DOI"),
            "journal": (it.get("container-title") or [""])[0],
            "year": safe_get(it,"issued","date-parts",default=[[None]])[0][0],
            "authors": [f"{a.get('family','')} {a.get('given','')}".strip() for a in (it.get("author") or [])],
            "source": "crossref"
        })
    return out

def openalex_search(intents: Dict[str,Any]) -> List[Dict]:
    kw = " ".join((intents.get("keywords") or [])[:8]) or " ".join(intents.get("topics") or [])
    if not kw: return []
    year_from = intents.get("year_from")
    year_to   = intents.get("year_to")
    params = {
        "search": kw,
        "from_publication_date": f"{year_from}-01-01",
        "to_publication_date": f"{year_to}-12-31",
        "per_page": OA_PER_PAGE
    }
    r = requests.get(OALEX_URL, params=params, timeout=60)
    if r.status_code != 200: return []
    out = []
    for it in r.json().get("results", []):
        doi = None
        ids = it.get("ids") or {}
        if "doi" in ids:
            doi = ids["doi"].replace("https://doi.org/","").lower()
        out.append({
            "title": it.get("title"),
            "doi": doi,
            "journal": (it.get("host_venue") or {}).get("display_name",""),
            "year": it.get("publication_year"),
            "authors": [a.get("author","").get("display_name","") for a in (it.get("authorships") or [])],
            "source": "openalex"
        })
    return out

# ============ 3) 打分 ============
def score_candidate(paragraph: str, intents: Dict[str,Any], item: Dict[str,Any]) -> float:
    p = paragraph.lower()
    title = (item.get("title") or "").lower()
    journal = (item.get("journal") or "").lower()
    try:
        year = int(re.findall(r"\d{4}", str(item.get("year") or ""))[0])
    except:
        year = 0

    kws = [k.lower() for k in (intents.get("keywords") or [])]
    kw_hits = sum(1 for k in kws if k in p or k in title)

    topics = [t.lower() for t in (intents.get("topics") or [])]
    topic_hits = sum(1 for t in topics if t in p or t in title)

    j_bonus = 1.0 if any(wj in journal for wj in WL_JOURNALS) else 0.0

    year_to = intents.get("year_to") or NOW_YEAR
    if year > 0:
        age = max(0, year_to - year)
        recency = 1.0 / (1.0 + 0.15*age)
    else:
        recency = 0.6

    title_overlap = 1.0 if any(w in title for w in kws[:3]) else 0.0

    return 2.0*kw_hits + 1.5*topic_hits + 1.2*j_bonus + 1.0*title_overlap + 2.0*recency

# ============ 3b) 获取论文摘要 ============
def fetch_abstract(doi: str) -> str:
    """Fetch abstract from Crossref API using DOI."""
    if not doi:
        return ""
    try:
        url = f"{CR_URL}{doi}"
        headers = {"User-Agent": f"cite-matcher (mailto:{CR_MAILTO})"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json().get("message", {})
            abstract = data.get("abstract", "")
            return abstract if abstract else ""
    except Exception:
        pass
    return ""


def load_kimi_cache() -> Dict[str, float]:
    try:
        if KIMI_CACHE_PATH.exists():
            return json.loads(KIMI_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_kimi_cache(cache: Dict[str, float]):
    try:
        KIMI_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"Failed to save Kimi cache: {e}", file=sys.stderr)


def merge_load_kimi_cache(defaults: Dict[str, float] = None) -> Dict[str, float]:
    """Load persistent Kimi cache and merge with provided defaults (if any)."""
    cache = load_kimi_cache()
    if defaults:
        for k, v in defaults.items():
            if k not in cache:
                cache[k] = v
    return cache

# ============ 3c) Kimi 相关性评分 ============
def kimi_score_relevance(paragraph: str, item: Dict[str,Any], intents: Dict[str,Any]) -> float:
    """Use Kimi to score paper relevance based on title + abstract against input paragraph."""
    title = norm(item.get("title"))
    abstract = norm(item.get("abstract", ""))
    
    if not title:
        return 0.0
    
    # If no abstract, use Kimi to score just the title
    paper_info = f"Title: {title}"
    if abstract:
        paper_info += f"\n\nAbstract: {abstract[:500]}"  # Limit abstract length to save tokens
    
    try:
        extractor = KimiExtractor(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, model=LLM_MODEL)
        score_prompt = f"""
You are evaluating the relevance of a paper to a research context.

Research paragraph (your context):
{paragraph}

Paper information:
{paper_info}

Rate the relevance on a scale of 0-10, where:
- 0-2: Not relevant
- 3-4: Loosely related
- 5-6: Moderately relevant
- 7-8: Highly relevant
- 9-10: Directly relevant

Return ONLY a JSON object with a single field "relevance_score" (integer from 0-10). No other text.
"""
        score_response = extractor.client.chat.completions.create(
            model=extractor.model,
            messages=[
                {"role": "user", "content": score_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        score_content = score_response.choices[0].message.content
        score_json = json.loads(score_content)
        score = float(score_json.get("relevance_score", 0))
        # Normalize to 0-2 scale to match with other scoring components
        return score / 5.0
    except Exception as e:
        print(f"Kimi scoring failed for DOI {item.get('doi')}: {e}", file=sys.stderr)
        return 0.0

# ============ 4) Unpaywall 标注 OA ============
def unpaywall_oa(doi: str) -> Dict[str,Any]:
    if not doi: return {"is_oa": False}
    url = f"{UNPAY_URL}{requests.utils.quote(doi)}?email={UNPAY_EMAIL}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200: return {"is_oa": False}
        j = r.json()
        best = None
        for loc in (j.get("oa_locations") or []):
            if loc.get("url_for_pdf"):
                best = loc.get("url_for_pdf")
                break
        if not best and (j.get("oa_locations") or []):
            best = j["oa_locations"][0].get("url")
        return {"is_oa": bool(j.get("is_oa")), "oa_url": best}
    except Exception:
        return {"is_oa": False}

# ============ 5) 引文格式 ============
def build_bibtex(item: Dict[str,Any]) -> str:
    doi = norm(item.get("doi"))
    title = norm(item.get("title"))
    journal = norm(item.get("journal"))
    year = norm(str(item.get("year") or ""))
    authors = [a for a in (item.get("authors") or []) if a]
    first_family = re.sub(r"[^A-Za-z0-9]+","", (authors[0].split()[-1] if authors else "anon"))
    jtag = re.sub(r"[^A-Za-z0-9]+","", journal[:6] or "J")
    citekey = f"{first_family}{year or 'XXXX'}{jtag}"
    author_field = " and ".join(authors)
    bib = textwrap.dedent(f"""
    @article{{{citekey}}},
      title={{ {title} }},
      author={{ {author_field} }},
      journal={{ {journal} }},
      year={{ {year} }},
      doi={{ {doi} }}
    }}
    """).strip()
    return bib

def build_csl_json(item: Dict[str,Any]) -> Dict[str,Any]:
    authors = []
    for a in (item.get("authors") or []):
        parts = a.split()
        if len(parts)>=2:
            authors.append({"family": parts[-1], "given": " ".join(parts[:-1])})
        else:
            authors.append({"literal": a})
    return {
        "type": "article-journal",
        "title": item.get("title"),
        "DOI": item.get("doi"),
        "issued": {"date-parts": [[int(item.get("year"))]]} if item.get("year") else None,
        "container-title": item.get("journal"),
        "author": authors
    }

def build_ris(item: Dict[str,Any]) -> str:
    """Build RIS format (RefMan) citation."""
    ris_lines = []
    ris_lines.append("TY  - JOUR")  # Type: Journal Article
    
    title = norm(item.get("title"))
    if title:
        ris_lines.append(f"TI  - {title}")
    
    # Authors
    authors = [a for a in (item.get("authors") or []) if a]
    for author in authors:
        ris_lines.append(f"AU  - {author}")
    
    journal = norm(item.get("journal"))
    if journal:
        ris_lines.append(f"JO  - {journal}")
    
    year = norm(str(item.get("year") or ""))
    if year:
        ris_lines.append(f"PY  - {year}")
    
    doi = norm(item.get("doi"))
    if doi:
        ris_lines.append(f"DO  - {doi}")
    
    source = item.get("source", "")
    if source:
        ris_lines.append(f"DB  - {source}")
    
    is_oa = item.get("is_oa", False)
    oa_url = item.get("oa_url", "")
    if is_oa and oa_url:
        ris_lines.append(f"UR  - {oa_url}")
    
    ris_lines.append("ER  - ")
    return "\n".join(ris_lines)

# ============ 6) 主流程 ============
def find_citations(paragraph: str) -> List[Dict[str,Any]]:
    intents = extract_intents(paragraph)

    pool = []
    pool += wos_search(intents)
    pool += crossref_search(intents)
    pool += openalex_search(intents)
    pool = dedup_by_doi(pool)

    # 1) First pass: fast traditional scoring (no external LLM calls)
    for it in pool:
        it["score"] = score_candidate(paragraph, intents, it)

    # 2) Sort and select a smaller candidate set for Kimi scoring to avoid rate limits
    pool.sort(key=lambda x: x["score"], reverse=True)
    kimi_eval_count = min(TOP_N * MAX_KIMI_MULTIPLIER, len(pool))
    top_candidates = pool[:kimi_eval_count]
    lower_candidates = pool[kimi_eval_count:]

    # 3) Fetch abstracts only for top candidates
    print(f"Fetching abstracts for top {kimi_eval_count} candidates...", file=sys.stderr)
    for it in top_candidates:
        doi = it.get("doi")
        if doi:
            it["abstract"] = fetch_abstract(doi)
        else:
            it["abstract"] = ""

    # 4) Kimi scoring with persistent caching and rate control
    kimi_cache: Dict[str, float] = merge_load_kimi_cache({})
    if not KIMI_ENABLED:
        print("Kimi scoring disabled by configuration/CLI; skipping LLM calls.", file=sys.stderr)
        # Keep traditional scores
        for idx, it in enumerate(top_candidates):
            it["score"] = it.get("score", 0.0)
    else:
        print(f"Kimi scoring top {kimi_eval_count} candidates (rate-controlled, persistent cache)...", file=sys.stderr)
        for idx, it in enumerate(top_candidates):
            doi = (it.get("doi") or "").lower()
            traditional = it.get("score", 0.0)
            kimi_score = 0.0
            try:
                if doi and doi in kimi_cache:
                    kimi_score = kimi_cache[doi]
                else:
                    kimi_score = kimi_score_relevance(paragraph, it, intents)
                    if doi:
                        kimi_cache[doi] = kimi_score
                        # persist intermittently to avoid losing work on long runs
                        if len(kimi_cache) % 10 == 0:
                            save_kimi_cache(kimi_cache)
            except Exception as e:
                print(f"Kimi scoring error for {doi}: {e}", file=sys.stderr)
                kimi_score = 0.0
            # Combine scores: 70% traditional, 30% Kimi-based
            it["score"] = 0.7 * traditional + 0.3 * kimi_score
            # small progress log
            print(f"[{idx+1}/{kimi_eval_count}] combined_score={it['score']:.3f}", file=sys.stderr)
        # save final cache state
        try:
            save_kimi_cache(kimi_cache)
        except Exception:
            pass

    # 5) Merge back and compute OA status for all candidates
    pool = top_candidates + lower_candidates
    print("Checking open access status...", file=sys.stderr)
    for it in pool:
        it.update(unpaywall_oa(it.get("doi","")))

    # 6) Final sort and output
    pool.sort(key=lambda x: x["score"], reverse=True)
    top = pool[:TOP_N]

    with open(OUT_JSONL, "a", encoding="utf-8") as fj, open(OUT_BIB, "a", encoding="utf-8") as fb, open(OUT_RIS, "a", encoding="utf-8") as fr:
        for it in top:
            it["bibtex"]   = build_bibtex(it)
            it["csl_json"] = build_csl_json(it)
            it["ris"]      = build_ris(it)
            fj.write(json.dumps(it, ensure_ascii=False) + "\n")
            fb.write("\n\n" + it["bibtex"] + "\n")
            fr.write("\n" + it["ris"] + "\n")

    return top

# ============ CLI ============
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="根据段落自动匹配参考文献（Kimi + WOS + Crossref + OpenAlex）")
    ap.add_argument("--text", help="直接传入一段文本", type=str)
    ap.add_argument("--file", help="从文件读取段落", type=str)
    ap.add_argument("--no-kimi", help="Disable Kimi/LLM scoring (fast-only mode)", action="store_true")
    ap.add_argument("--kimi-cache", help="Path to persistent Kimi cache (overrides config)", type=str)
    ap.add_argument("--kimi-multiplier", help="Multiplier for how many top candidates receive Kimi scoring (default from config)", type=int)
    args = ap.parse_args()

    if args.file:
        paragraph = open(args.file, "r", encoding="utf-8").read()
    elif args.text:
        paragraph = args.text
    else:
        print("请用 --text 或 --file 传入段落。")
        raise SystemExit(1)

    # Apply CLI overrides for Kimi behavior
    if args.no_kimi:
        KIMI_ENABLED = False
        print("Kimi/LLM scoring disabled via --no-kimi", file=sys.stderr)
    if args.kimi_cache:
        try:
            KIMI_CACHE_PATH = pathlib.Path(args.kimi_cache)
            print(f"Using Kimi cache path: {KIMI_CACHE_PATH}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to set KIMI_CACHE_PATH from CLI: {e}", file=sys.stderr)
    if args.kimi_multiplier:
        try:
            MAX_KIMI_MULTIPLIER = int(args.kimi_multiplier)
            print(f"Kimi multiplier set to {MAX_KIMI_MULTIPLIER}", file=sys.stderr)
        except Exception as e:
            print(f"Invalid --kimi-multiplier value: {e}", file=sys.stderr)

    res = find_citations(paragraph)
    print(f"Found {len(res)} candidates (Top-{TOP_N}). 输出已写入：\n- {OUT_JSONL}\n- {OUT_BIB}\n- {OUT_RIS}")
    for i, it in enumerate(res, 1):
        print(f"{i}. {it.get('title')} ({it.get('journal')}, {it.get('year')}) doi:{it.get('doi')}  OA:{it.get('is_oa')}")


