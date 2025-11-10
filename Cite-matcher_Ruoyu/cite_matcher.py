# cite_matcher.py auto-matches citations and builds bibliographic entries
import os, re, json, time, math, yaml, pathlib, requests, argparse, textwrap
from datetime import datetime
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

CR_URL       = CFG["crossref"]["works_endpoint"]
CR_MAILTO    = CFG["crossref"]["polite_mailto"]
OALEX_URL    = CFG["openalex"]["works_endpoint"]

UNPAY_URL    = CFG["unpaywall"]["endpoint"]
UNPAY_EMAIL  = CFG["unpaywall"]["email"]

OUT_DIR      = pathlib.Path(CFG["output"]["out_dir"]); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL    = pathlib.Path(CFG["output"]["results_json"])
OUT_BIB      = pathlib.Path(CFG["output"]["bibtex_path"])
TOP_N        = int(CFG["output"]["top_n"])

NOW_YEAR     = datetime.utcnow().year

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
    intents["keywords"] = [k for k in (intents.get("keywords") or []) if len(k.strip())>1][:10]
    intents["topics"]   = intents.get("topics") or []
    return intents

# ============ 1) WOS 检索 ============
def wos_search(intents: Dict[str,Any]) -> List[Dict]:
    assert WOS_KEY, "Missing WOS API key in config.yaml"
    kws = intents.get("keywords") or intents.get("topics") or []
    if not kws:
        return []
    q = " OR ".join([f"TS=({k})" for k in kws])
    year_from = intents.get("year_from")
    year_to   = intents.get("year_to")

    params = {
        "q": q,
        "from": 0,
        "limit": 50,
        "sort": "load_date:desc",
        "publication_year_from": year_from,
        "publication_year_to": year_to
    }
    headers = {"X-ApiKey": WOS_KEY, "Accept": "application/json"}
    out = []
    while True:
        r = requests.get(WOS_URL, params=params, headers=headers, timeout=60)
        r.raise_for_status()
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
    params = {"query": kw, "filter": f"from-pub-date:{year_from}-01-01,until-pub-date:{year_to}-12-31", "rows": 50}
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
        "per_page": 50
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

# ============ 6) 主流程 ============
def find_citations(paragraph: str) -> List[Dict[str,Any]]:
    intents = extract_intents(paragraph)

    pool = []
    pool += wos_search(intents)
    pool += crossref_search(intents)
    pool += openalex_search(intents)
    pool = dedup_by_doi(pool)

    for it in pool:
        it["score"] = score_candidate(paragraph, intents, it)
    for it in pool:
        it.update(unpaywall_oa(it.get("doi","")))

    pool.sort(key=lambda x: x["score"], reverse=True)
    top = pool[:TOP_N]

    # 输出
    with open(OUT_JSONL, "a", encoding="utf-8") as fj, open(OUT_BIB, "a", encoding="utf-8") as fb:
        for it in top:
            it["bibtex"]   = build_bibtex(it)
            it["csl_json"] = build_csl_json(it)
            fj.write(json.dumps(it, ensure_ascii=False) + "\n")
            fb.write("\n\n" + it["bibtex"] + "\n")

    return top

# ============ CLI ============
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="根据段落自动匹配参考文献（Kimi + WOS + Crossref + OpenAlex）")
    ap.add_argument("--text", help="直接传入一段文本", type=str)
    ap.add_argument("--file", help="从文件读取段落", type=str)
    args = ap.parse_args()

    if args.file:
        paragraph = open(args.file, "r", encoding="utf-8").read()
    elif args.text:
        paragraph = args.text
    else:
        print("请用 --text 或 --file 传入段落。")
        raise SystemExit(1)

    res = find_citations(paragraph)
    print(f"Found {len(res)} candidates (Top-{TOP_N}). 输出已写入：\n- {OUT_JSONL}\n- {OUT_BIB}")
    for i, it in enumerate(res, 1):
        print(f"{i}. {it.get('title')} ({it.get('journal')}, {it.get('year')}) doi:{it.get('doi')}  OA:{it.get('is_oa')}")


