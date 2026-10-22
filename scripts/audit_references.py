#!/usr/bin/env python3
"""Resolve manuscript DOI metadata from Crossref without guessing authors."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

REFERENCES = {
    "cinelli2020": "10.1016/j.omega.2020.102261",
    "greco2025": "10.1016/j.ejor.2024.07.038",
    "fu2021": "10.1016/j.asoc.2021.107622",
    "maghsoodi2023": "10.1016/j.engappai.2022.105785",
    "zindani2025": "10.1186/s44147-025-00789-4",
    "wang2026": "10.3390/a19090762",
    "upadhyay2026": "10.1016/j.mex.2026.104007",
    "hooker2021": "10.1007/s11222-021-10057-z",
    "critic1995": "10.1016/0305-0548(94)00059-H",
    "merec2021": "10.3390/sym13040525",
    "fucom2018": "10.3390/sym10090393",
    "tervonen2013": "10.1016/j.ejor.2012.08.026",
    "corrente2017": "10.1016/j.omega.2016.11.008",
    "breiman2001": "10.1023/A:1010933404324",
    "zou2005": "10.1111/j.1467-9868.2005.00503.x",
    "geurts2006": "10.1007/s10994-006-6226-1",
    "friedman2001": "10.1214/aos/1013203451",
    "varma2006": "10.1186/1471-2105-7-91",
    "altmann2010": "10.1093/bioinformatics/btq134",
    "waspas2012": "10.5755/j01.eee.122.6.1810",
    "marcos2020": "10.1016/j.cie.2019.106231",
    "shannon1948": "10.1002/j.1538-7305.1948.tb01338.x",
    "kizielewicz2024": "10.1016/j.procs.2024.09.673",
}

def resolve(item):
    key, doi = item
    request = Request("https://api.crossref.org/works/" + quote(doi, safe=""),
                      headers={"User-Agent": "TechPort-reference-audit/3.0"})
    try:
        with urlopen(request, timeout=45) as response:
            m = json.load(response)["message"]
        return key, dict(doi=doi, title=m.get("title", []), authors=m.get("author", []),
                         journal=m.get("container-title", []), volume=m.get("volume"),
                         issue=m.get("issue"), page=m.get("page"),
                         article_number=m.get("article-number"),
                         published=m.get("published"), published_print=m.get("published-print"),
                         published_online=m.get("published-online"), url=m.get("URL"),
                         checked="2026-10-22", status="resolved")
    except Exception as e:
        return key, dict(doi=doi, status="unresolved", error=str(e))

if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "manuscript" / "reference_audit.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    missing = [(k, d) for k, d in REFERENCES.items() if result.get(k, {}).get("status") != "resolved"]
    result.update(dict(ThreadPoolExecutor(max_workers=2).map(resolve, missing)))
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {"status": v["status"], "authors": [a.get("family") for a in v.get("authors", [])],
                         "title": v.get("title"), "published": v.get("published")}
                      for k, v in result.items()}, ensure_ascii=False, indent=2))
