#!/usr/bin/env python3
"""Generate CBDB 20260516 science/technical personnel network files.

This module extracts technical/scientific officials and political elites from a
CBDB SQLite database snapshot. It computes network paths via BFS, formats node
and edge data, validates graph integrity, and generates both a raw JSON dataset
and an interactive D3.js visualization HTML file.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

# Base path definitions
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT.parent / "latest" / "cbdb_20260516.sqlite3"
JSON_OUT = ROOT / "science_network_data_20260516.json"
HTML_OUT = ROOT / "network_20260516.html"
D3_SOURCE = ROOT / "d3.v7.min.js"

# Algorithmic pathfinding limits
MAX_HIGH_PATH_DISTANCE = 6
MAX_HIGH_PATHS_PER_MAIN = 8

# Target keywords used to identify technical/scientific bureaucracy positions
TECHNICAL_KEYWORDS = [
    "工部", "司天", "欽天", "钦天", "太醫", "太医", "都水", "軍器", "军器", "將作", "将作",
    "天文", "算學", "算学", "算曆", "算历", "曆算", "历算", "造曆", "造历",
    "醫官", "医官", "醫學", "医学", "醫正", "医正", "醫生", "医生", "醫士", "医士",
    "醫人", "医人", "御醫", "御医", "翰林醫", "翰林医", "翰林天文", "河道", "水利",
    "繕工", "缮工", "百工", "營繕", "营缮",
]
TECHNICAL_RE = re.compile("|".join(re.escape(k) for k in TECHNICAL_KEYWORDS))

# Department groupings for categorizing technical office roles and color coding
BUREAU_GROUPS = [
    {"key": "astronomical", "zh": "天文/历算", "en": "Astronomical", "color": "#6f8fa6", "pattern": re.compile(r"司天|欽天|钦天|天文|算學|算学|算曆|算历|曆算|历算|造曆|造历|推算|靈臺|灵台")},
    {"key": "medical", "zh": "医学/医官", "en": "Medical", "color": "#b9828c", "pattern": re.compile(r"太醫|太医|醫官|医官|醫學|医学|醫正|医正|醫生|医生|醫士|医士|醫人|医人|御醫|御医|翰林醫|翰林医")},
    {"key": "hydraulic", "zh": "水利/河道", "en": "Hydraulic", "color": "#7fa59a", "pattern": re.compile(r"都水|河道|水利")},
    {"key": "military_industrial", "zh": "军器/制造", "en": "Military-industrial", "color": "#c08a6a", "pattern": re.compile(r"軍器|军器|兵器|火器|器械")},
    {"key": "construction", "zh": "将作/营缮", "en": "Construction", "color": "#b6a27a", "pattern": re.compile(r"將作|将作|繕工|缮工|百工|營繕|营缮|修造|土木")},
    {"key": "works_general", "zh": "工部", "en": "Ministry of Works", "color": "#9a8fb2", "pattern": re.compile(r"工部")},
    {"key": "other_technical", "zh": "其他技术官职", "en": "Other technical", "color": "#8fa37a", "pattern": re.compile(r".*")},
]

# High official keyword filters to isolate key political elites
HIGH_OFFICIAL_KEYWORDS = [
    "同中書門下二品", "同中书门下二品", "同中書門下三品", "同中书门下三品",
    "同中書門下平章事", "同中书门下平章事", "同鳳閣鸞臺", "同凤阁鸾台",
    "平章事", "中書令", "中书令", "尚書令", "尚书令", "侍中", "門下侍郎", "门下侍郎",
    "丞相", "宰相", "參知政事", "参知政事", "內閣大學士", "内阁大学士", "內閣大學士", "内阁大學士",
    "軍機大臣", "军机大臣",
]
HIGH_OFFICIAL_RE = re.compile("|".join(re.escape(k) for k in HIGH_OFFICIAL_KEYWORDS))

# Exclusion criteria to filter out clerical/subordinate roles matching high official patterns
HIGH_OFFICIAL_EXCLUDE_RE = re.compile(
    r"內閣中書|内阁中书|中書舍人|中书舍人|中書省|中书省|中書檢正|中书检正|"
    r"門下省|门下省|中書五房|中书五房|書令史|令史|習學|习学|孔目|吏房|戶房|户房|"
    r"禮房|礼房|刑房|兵房|工房|開拆|章奏|催驅|点檢|點檢"
)

# Mandatory database tables required for graph extraction
REQUIRED_TABLES = {"BIOG_MAIN", "OFFICE_CODES", "POSTED_TO_OFFICE_DATA", "KIN_DATA", "ASSOC_DATA", "ASSOC_CODES", "KINSHIP_CODES", "DYNASTIES"}


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def group_for_office(office_zh: str) -> dict[str, Any]:
    for group in BUREAU_GROUPS:
        if group["pattern"].search(office_zh):
            return group
    return BUREAU_GROUPS[-1]


def validate_schema(conn: sqlite3.Connection) -> None:
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    missing = sorted(REQUIRED_TABLES - tables)
    if missing:
        raise RuntimeError(f"Missing required tables: {', '.join(missing)}")


def load_dynasties(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    return {
        int(row["c_dy"]): {
            "code": str(int(row["c_dy"])),
            "zh": clean(row["c_dynasty_chn"]),
            "en": clean(row["c_dynasty"]),
            "sort": int_or_none(row["c_sort"]) or 999,
        }
        for row in conn.execute("select c_dy, c_dynasty, c_dynasty_chn, c_sort from DYNASTIES")
    }


def load_people(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    people: dict[int, dict[str, Any]] = {}
    for row in conn.execute("select c_personid, c_name, c_name_chn, c_index_year, c_dy, c_female, c_ethnicity_code from BIOG_MAIN"):
        pid = int(row["c_personid"])
        people[pid] = {
            "id": pid,
            "nameZh": clean(row["c_name_chn"]) or clean(row["c_name"]) or str(pid),
            "nameEn": clean(row["c_name"]) or clean(row["c_name_chn"]) or str(pid),
            "indexYear": int_or_none(row["c_index_year"]),
            "dynastyCode": int_or_none(row["c_dy"]),
            "female": int_or_none(row["c_female"]),
            "ethnicityCode": int_or_none(row["c_ethnicity_code"]),
        }
    return people


def load_relation_labels(conn: sqlite3.Connection) -> tuple[dict[int, str], dict[int, str]]:
    assoc = {int(r["c_assoc_code"]): clean(r["c_assoc_desc"]) or clean(r["c_assoc_desc_chn"]) or "association" for r in conn.execute("select c_assoc_code, c_assoc_desc, c_assoc_desc_chn from ASSOC_CODES")}
    kin = {int(r["c_kincode"]): clean(r["c_kinrel"]) or clean(r["c_kinrel_chn"]) or "kinship" for r in conn.execute("select c_kincode, c_kinrel, c_kinrel_chn from KINSHIP_CODES")}
    return assoc, kin


def extract_technical_people(conn: sqlite3.Connection) -> tuple[set[int], set[int], set[int], dict[int, dict[int, dict[str, Any]]], Counter[str]]:
    all_offices: set[int] = set()
    used_offices: set[int] = set()
    offices_by_person: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    group_counts: Counter[str] = Counter()

    for row in conn.execute("select c_office_id, c_office_chn from OFFICE_CODES where c_office_chn is not null"):
        office_zh = clean(row["c_office_chn"])
        if TECHNICAL_RE.search(office_zh):
            all_offices.add(int(row["c_office_id"]))

    sql = """
        select p.c_personid, p.c_office_id, p.c_sequence, p.c_firstyear, p.c_lastyear,
               o.c_office_chn, o.c_office_pinyin, o.c_office_trans
        from POSTED_TO_OFFICE_DATA p
        join OFFICE_CODES o on p.c_office_id = o.c_office_id
        where p.c_personid is not null and o.c_office_chn is not null
    """
    for row in conn.execute(sql):
        office_zh = clean(row["c_office_chn"])
        if not TECHNICAL_RE.search(office_zh):
            continue
        pid = int(row["c_personid"])
        oid = int(row["c_office_id"])
        group = group_for_office(office_zh)
        used_offices.add(oid)
        offices_by_person[pid][oid] = {
            "id": oid,
            "zh": office_zh,
            "pinyin": clean(row["c_office_pinyin"]),
            "en": clean(row["c_office_trans"]),
            "sequence": int_or_none(row["c_sequence"]),
            "firstYear": int_or_none(row["c_firstyear"]),
            "lastYear": int_or_none(row["c_lastyear"]),
            "bureauType": group["key"],
            "bureauZh": group["zh"],
            "bureauEn": group["en"],
        }

    for offices in offices_by_person.values():
        primary = sorted(offices.values(), key=lambda x: (x["sequence"] is None, x["sequence"] or 999999, x["id"]))[0]
        group_counts[primary["bureauType"]] += 1
    return all_offices, used_offices, set(offices_by_person), offices_by_person, group_counts


def extract_high_officials(conn: sqlite3.Connection) -> tuple[set[int], dict[int, dict[int, dict[str, Any]]], Counter[str]]:
    high_ids: set[int] = set()
    offices_by_person: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    office_counts: Counter[str] = Counter()
    sql = """
        select p.c_personid, p.c_office_id, p.c_sequence, p.c_firstyear, p.c_lastyear,
               o.c_office_chn, o.c_office_pinyin, o.c_office_trans
        from POSTED_TO_OFFICE_DATA p
        join OFFICE_CODES o on p.c_office_id = o.c_office_id
        where p.c_personid is not null and o.c_office_chn is not null
    """
    for row in conn.execute(sql):
        office_zh = clean(row["c_office_chn"])
        if not HIGH_OFFICIAL_RE.search(office_zh) or HIGH_OFFICIAL_EXCLUDE_RE.search(office_zh):
            continue
        pid = int(row["c_personid"])
        oid = int(row["c_office_id"])
        high_ids.add(pid)
        office_counts[office_zh] += 1
        offices_by_person[pid][oid] = {
            "id": oid,
            "zh": office_zh,
            "pinyin": clean(row["c_office_pinyin"]),
            "en": clean(row["c_office_trans"]),
            "sequence": int_or_none(row["c_sequence"]),
            "firstYear": int_or_none(row["c_firstyear"]),
            "lastYear": int_or_none(row["c_lastyear"]),
        }
    return high_ids, offices_by_person, office_counts


def build_adjacency(conn: sqlite3.Connection, assoc_labels: dict[int, str], kin_labels: dict[int, str]) -> tuple[defaultdict[int, set[int]], dict[tuple[int, int], dict[str, Any]]]:
    adjacency: defaultdict[int, set[int]] = defaultdict(set)
    edge_relation: dict[tuple[int, int], dict[str, Any]] = {}

    def add_edge(a: Any, b: Any, kind: str, code: Any, label: str) -> None:
        ai = int_or_none(a)
        bi = int_or_none(b)
        if ai is None or bi is None or ai == bi:
            return
        adjacency[ai].add(bi)
        adjacency[bi].add(ai)
        key = tuple(sorted((ai, bi)))
        edge_relation.setdefault(key, {"source": key[0], "target": key[1], "kind": kind, "code": int_or_none(code), "type": clean(label) or kind})

    for row in conn.execute("select c_personid, c_assoc_id, c_assoc_code from ASSOC_DATA where c_personid is not null and c_assoc_id is not null"):
        code = int_or_none(row["c_assoc_code"])
        add_edge(row["c_personid"], row["c_assoc_id"], "assoc", code, assoc_labels.get(code or -1, "association"))
    for row in conn.execute("select c_personid, c_kin_id, c_kin_code from KIN_DATA where c_personid is not null and c_kin_id is not null"):
        code = int_or_none(row["c_kin_code"])
        add_edge(row["c_personid"], row["c_kin_id"], "kin", code, kin_labels.get(code or -1, "kinship"))
    return adjacency, edge_relation


def find_high_paths(main_ids: set[int], high_ids: set[int], adjacency: dict[int, set[int]], people: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], set[tuple[int, int]], set[int], set[int], set[int]]:
    paths: list[dict[str, Any]] = []
    path_pairs: set[tuple[int, int]] = set()
    path_node_ids: set[int] = set()
    covered_main_ids: set[int] = set()
    related_high_ids: set[int] = set()

    for main_id in sorted(main_ids):
        queue: deque[tuple[int, list[int]]] = deque([(main_id, [main_id])])
        seen = {main_id}
        found = 0
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= MAX_HIGH_PATH_DISTANCE:
                continue
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                next_path = path + [neighbor]
                distance = len(next_path) - 1
                if neighbor in high_ids and neighbor != main_id:
                    found += 1
                    covered_main_ids.add(main_id)
                    related_high_ids.add(neighbor)
                    path_node_ids.update(next_path)
                    for source, target in zip(next_path, next_path[1:]):
                        path_pairs.add(tuple(sorted((source, target))))
                    paths.append({
                        "mainId": main_id,
                        "highId": neighbor,
                        "distance": distance,
                        "intermediateCount": max(0, distance - 1),
                        "pathIds": next_path,
                        "pathNames": [people.get(pid, {}).get("nameZh") or people.get(pid, {}).get("nameEn") or str(pid) for pid in next_path],
                    })
                    if found >= MAX_HIGH_PATHS_PER_MAIN:
                        break
                if distance < MAX_HIGH_PATH_DISTANCE:
                    queue.append((neighbor, next_path))
            if found >= MAX_HIGH_PATHS_PER_MAIN:
                break
    return paths, path_pairs, path_node_ids, covered_main_ids, related_high_ids


def sample_technical_offices(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = {group["key"]: [] for group in BUREAU_GROUPS}
    for row in conn.execute("select c_office_id, c_office_chn, c_office_trans from OFFICE_CODES where c_office_chn is not null order by c_office_id"):
        office_zh = clean(row["c_office_chn"])
        if not TECHNICAL_RE.search(office_zh):
            continue
        group = group_for_office(office_zh)
        if len(samples[group["key"]]) < 8:
            samples[group["key"]].append({"id": int(row["c_office_id"]), "zh": office_zh, "en": clean(row["c_office_trans"])})
    return samples


def build_data() -> tuple[dict[str, Any], dict[str, Any]]:
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    validate_schema(conn)

    dynasties = load_dynasties(conn)
    people = load_people(conn)
    assoc_labels, kin_labels = load_relation_labels(conn)
    all_technical_offices, used_technical_offices, main_ids, offices_by_person, group_counts = extract_technical_people(conn)
    high_ids_all, high_offices_by_person, high_office_counts = extract_high_officials(conn)
    adjacency, edge_relation = build_adjacency(conn, assoc_labels, kin_labels)

    direct_pairs: set[tuple[int, int]] = set()
    direct_neighbors: set[int] = set()
    for main_id in main_ids:
        for neighbor in adjacency.get(main_id, ()):
            direct_pairs.add(tuple(sorted((main_id, neighbor))))
            direct_neighbors.add(neighbor)

    paths, path_pairs, path_node_ids, covered_main_ids, related_high_ids = find_high_paths(main_ids, high_ids_all, adjacency, people)
    bridge_ids = {pid for pid in path_node_ids if pid not in main_ids and pid not in related_high_ids}
    direct_person_ids = {pid for pid in direct_neighbors if pid not in main_ids and pid not in related_high_ids and pid not in bridge_ids}

    node_ids = set(main_ids) | direct_person_ids | bridge_ids | related_high_ids
    for source, target in direct_pairs | path_pairs:
        node_ids.add(source)
        node_ids.add(target)

    def node_type(pid: int) -> str:
        if pid in main_ids:
            return "main"
        if pid in related_high_ids:
            return "high"
        if pid in bridge_ids:
            return "bridge"
        return "other"

    def make_node(pid: int) -> dict[str, Any]:
        person = people.get(pid, {"id": pid, "nameZh": str(pid), "nameEn": str(pid), "dynastyCode": None})
        dynasty_code = person.get("dynastyCode")
        dynasty = dynasties.get(dynasty_code, {"code": str(dynasty_code or ""), "zh": "", "en": "", "sort": 999})
        technical_offices = sorted(offices_by_person.get(pid, {}).values(), key=lambda item: (item.get("sequence") is None, item.get("sequence") or 999999, item["id"]))
        primary_group = group_for_office(technical_offices[0]["zh"]) if technical_offices else BUREAU_GROUPS[-1]
        return {
            "id": pid,
            "nameZh": person.get("nameZh") or person.get("nameEn") or str(pid),
            "nameEn": person.get("nameEn") or person.get("nameZh") or str(pid),
            "indexYear": person.get("indexYear"),
            "dynastyCode": dynasty_code,
            "dynastyZh": dynasty["zh"],
            "dynastyEn": dynasty["en"],
            "dynastySort": dynasty["sort"],
            "nodeType": node_type(pid),
            "isMain": pid in main_ids,
            "isHighOfficial": pid in related_high_ids,
            "isBridge": pid in bridge_ids,
            "bureauType": primary_group["key"] if pid in main_ids else "",
            "bureauZh": primary_group["zh"] if pid in main_ids else "",
            "bureauEn": primary_group["en"] if pid in main_ids else "",
            "scienceOffices": technical_offices,
            "highOffices": sorted(high_offices_by_person.get(pid, {}).values(), key=lambda item: (item.get("sequence") is None, item.get("sequence") or 999999, item["id"])),
        }

    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[Any, ...]] = set()

    def add_output_edge(pair: tuple[int, int], category: str) -> None:
        relation = dict(edge_relation.get(pair, {"source": pair[0], "target": pair[1], "kind": "relation", "code": None, "type": "relation"}))
        relation["category"] = category
        key = (relation["source"], relation["target"], relation["kind"], relation["code"], relation["category"])
        if key not in edge_keys:
            edge_keys.add(key)
            edges.append(relation)

    for pair in sorted(direct_pairs):
        add_output_edge(pair, "direct")
    for pair in sorted(path_pairs):
        add_output_edge(pair, "path")

    nodes = sorted((make_node(pid) for pid in node_ids), key=lambda n: (n.get("dynastySort") or 999, n["nodeType"], n["id"]))
    used_dynasty_codes = {n["dynastyCode"] for n in nodes if n.get("dynastyCode") is not None}
    dynasty_options = sorted((d for code, d in dynasties.items() if code in used_dynasty_codes), key=lambda d: (d["sort"], int(d["code"]) if str(d["code"]).isdigit() else 9999))
    main_direct_coverage = {pid for pid in main_ids if any(pid in pair for pair in direct_pairs)}

    summary = {
        "dbPath": str(DB_PATH),
        "scienceOfficeCount": len(all_technical_offices),
        "usedScienceOfficeCount": len(used_technical_offices),
        "mainPersonCount": len(main_ids),
        "highOfficialCandidateCount": len(high_ids_all),
        "relatedHighOfficialCount": len(related_high_ids),
        "directPersonCount": len(direct_person_ids),
        "bridgeCount": len(bridge_ids),
        "nodeCount": len(nodes),
        "edgeCount": len(edges),
        "directEdgeCount": sum(1 for e in edges if e["category"] == "direct"),
        "pathEdgeCount": sum(1 for e in edges if e["category"] == "path"),
        "pathCount": len(paths),
        "mainWithDirectRelationsCount": len(main_direct_coverage),
        "highPathCoveredMainCount": len(covered_main_ids),
        "maxHighPathDistance": MAX_HIGH_PATH_DISTANCE,
        "maxHighPathsPerMain": MAX_HIGH_PATHS_PER_MAIN,
        "technicalKeywords": TECHNICAL_KEYWORDS,
        "highOfficialKeywords": HIGH_OFFICIAL_KEYWORDS,
        "bureauTypeCounts": dict(group_counts),
        "topHighOfficialOffices": high_office_counts.most_common(30),
    }

    data = {
        "summary": summary,
        "bureauGroups": [{"key": g["key"], "zh": g["zh"], "en": g["en"], "color": g["color"]} for g in BUREAU_GROUPS],
        "dynasties": dynasty_options,
        "nodes": nodes,
        "edges": edges,
        "paths": paths,
    }
    review = {
        "technicalOfficeSamples": sample_technical_offices(conn),
        "highOfficialOfficeSamples": high_office_counts.most_common(20),
        "nodeTypeCounts": Counter(n["nodeType"] for n in nodes),
    }
    conn.close()
    return data, review


def validate_data(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    node_ids = [n["id"] for n in data["nodes"]]
    node_set = set(node_ids)
    if len(node_ids) != len(node_set):
        errors.append("duplicate node ids")
    for edge in data["edges"]:
        if edge["source"] not in node_set or edge["target"] not in node_set:
            errors.append(f"missing edge endpoint: {edge}")
            break
    main_ids = {n["id"] for n in data["nodes"] if n["nodeType"] == "main"}
    high_ids = {n["id"] for n in data["nodes"] if n.get("isHighOfficial")}
    for path in data["paths"]:
        if path["mainId"] not in main_ids:
            errors.append(f"path main missing/not main: {path['mainId']}")
            break
        if path["highId"] not in high_ids:
            errors.append(f"path high missing/not high: {path['highId']}")
            break
        if any(pid not in node_set for pid in path["pathIds"]):
            errors.append(f"path node missing: {path}")
            break
    summary = data["summary"]
    if summary["nodeCount"] != len(data["nodes"]):
        errors.append("summary node count mismatch")
    if summary["edgeCount"] != len(data["edges"]):
        errors.append("summary edge count mismatch")
    if summary["pathCount"] != len(data["paths"]):
        errors.append("summary path count mismatch")
    return errors


def render_html(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    d3_source = D3_SOURCE.read_text() if D3_SOURCE.exists() else ""
    template = r'''<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CBDB Science and Technical Personnel Network</title>
  <script>__D3_SOURCE__</script>
  <style>
    :root { --bg:#edf3f8; --panel:rgba(255,255,255,.94); --ink:#1f2933; --muted:#677381; --line:#c9d2dc; --high:#b8abc8; --bridge:#f2cf63; --other:#a0a6a3; --path:#d7a63f; --direct:#8d9694; }
    * { box-sizing:border-box; } body { margin:0; overflow:hidden; color:var(--ink); font-family:Arial,"Noto Sans SC","Microsoft YaHei",sans-serif; background:var(--bg); }
    #network { width:100vw; height:100vh; background:linear-gradient(135deg,rgba(255,255,255,.76),rgba(217,229,242,.88)),radial-gradient(circle at 20% 12%,rgba(34,167,201,.12),transparent 28%),radial-gradient(circle at 88% 82%,rgba(216,198,255,.18),transparent 26%); }
    svg { display:block; }
    .topbar { position:fixed; top:10px; left:50%; transform:translateX(-50%); z-index:20; text-align:center; pointer-events:none; width:min(720px,52vw); min-width:430px; padding:7px 12px; border-radius:10px; background:rgba(238,243,248,.72); backdrop-filter:blur(5px); }
    .topbar h1 { margin:0; font-size:22px; line-height:1.05; } .topbar .subtitle { margin-top:3px; color:var(--muted); font-size:12px; }
    .controls { position:fixed; top:10px; right:10px; z-index:25; width:270px; max-height:calc(100vh - 20px); overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px; box-shadow:0 8px 22px rgba(31,41,51,.13); }
    .control-row { display:grid; gap:4px; margin-bottom:7px; } .control-row label { font-size:12px; color:#3c4652; }
    select,input[type="search"] { width:100%; border:1px solid #c3ccd6; background:#fff; color:var(--ink); border-radius:6px; padding:6px 7px; font-size:12px; outline:none; }
    .segmented { display:grid; grid-template-columns:repeat(3,1fr); gap:4px; } button.segment { border:1px solid #c3ccd6; background:#fff; border-radius:6px; padding:6px 4px; font-size:12px; cursor:pointer; } button.segment.active { background:#243447; color:#fff; border-color:#243447; }
    .checkgrid { display:grid; grid-template-columns:1fr 1fr; gap:6px 8px; } .checkgrid label { display:flex; gap:6px; align-items:center; font-size:12px; color:#2f3b47; }
    .status { border-top:1px solid #dce2e9; padding-top:8px; font-size:12px; color:var(--muted); line-height:1.45; }
    .legend { position:fixed; left:10px; bottom:10px; z-index:18; background:rgba(255,255,255,.9); border:1px solid var(--line); border-radius:8px; padding:8px 10px; min-width:190px; max-width:285px; font-size:11px; box-shadow:0 6px 18px rgba(31,41,51,.11); }
    .legend-title { margin:5px 0 4px; color:#344150; font-weight:800; font-size:11px; } .legend-row { display:flex; align-items:center; gap:7px; margin:5px 0; }
    .swatch { width:13px; height:13px; border-radius:50%; border:1px solid rgba(0,0,0,.15); flex:none; } .swatch-line { width:22px; height:0; border-top:2px solid var(--direct); flex:none; } .swatch-line.path { border-top:2px dashed var(--path); }
    .link { stroke:var(--direct); stroke-opacity:.42; stroke-width:1.1px; } .link.path { stroke:var(--path); stroke-dasharray:5 4; stroke-opacity:.76; stroke-width:1.45px; }
    .node { stroke:#fff; stroke-width:1.6px; cursor:pointer; } .node.main { stroke:#4f5858; stroke-opacity:.5; stroke-width:1px; } .node.high { stroke:#7d7190; stroke-width:1.8px; }
    .label { pointer-events:none; text-anchor:middle; fill:#26323f; font-weight:700; paint-order:stroke; stroke:rgba(238,243,248,.86); stroke-width:3px; }
    .tooltip { position:fixed; pointer-events:none; z-index:40; max-width:390px; padding:10px 12px; border-radius:8px; background:rgba(18,24,33,.94); color:#fff; font-size:13px; line-height:1.45; box-shadow:0 10px 28px rgba(0,0,0,.22); opacity:0; }
    .tooltip .muted { color:#c9d2dc; font-size:12px; margin-top:4px; } .empty { position:fixed; inset:auto 50% 24px auto; transform:translateX(50%); z-index:19; background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:8px; padding:8px 12px; color:var(--muted); font-size:13px; display:none; }
  </style>
</head>
<body>
  <div id="network"></div>
  <div class="topbar"><h1 id="title"></h1><div class="subtitle" id="subtitle"></div></div>
  <aside class="controls">
    <div class="control-row"><label id="languageLabel" for="language"></label><select id="language"><option value="zh">中文</option><option value="en">English</option></select></div>
    <div class="control-row"><label id="dynastyLabel" for="dynasty"></label><select id="dynasty"></select></div>
    <div class="control-row"><label id="viewLabel"></label><div class="segmented"><button class="segment" data-view="all" id="viewAll"></button><button class="segment" data-view="direct" id="viewDirect"></button><button class="segment active" data-view="paths" id="viewPaths"></button></div></div>
    <div class="control-row"><label id="searchLabel" for="search"></label><input id="search" type="search"></div>
    <div class="control-row"><label id="nodeTypeLabel"></label><div class="checkgrid"><label><input type="checkbox" class="type-toggle" value="main" checked><span id="toggleMain"></span></label><label><input type="checkbox" class="type-toggle" value="high" checked><span id="toggleHigh"></span></label><label><input type="checkbox" class="type-toggle" value="bridge" checked><span id="toggleBridge"></span></label><label><input type="checkbox" class="type-toggle" value="other" checked><span id="toggleOther"></span></label></div></div>
    <div class="control-row"><label><input id="labels" type="checkbox"> <span id="labelsText"></span></label></div>
    <div class="status" id="status"></div>
  </aside>
  <div class="legend" id="legend"></div><div class="empty" id="empty"></div><div class="tooltip" id="tooltip"></div>
  <script>
    const DATA = __DATA_JSON__;
    const i18n = {
      zh: { title:'CBDB 中国古代科学/技术人员关系网络', subtitle:s=>`${s.main} 位主技术人员 · ${s.high} 位相关高官 · ${s.other} 位直接人物 · ${s.bridge} 位中间人物`, language:'语言', dynasty:'朝代筛选', allDynasties:'全部朝代', view:'视图模式', all:'全图', direct:'直接关系', paths:'高官路径', search:'搜索人物', searchPlaceholder:'输入中文名、英文名或 ID', nodeType:'节点类型', main:'主技术人员', high:'高官', bridge:'中间人物', other:'直接人物', labels:'显示标签', bureauColor:'主技术人员官职颜色', relationNodes:'关系节点', directEdge:'直接关系', pathEdge:'高官路径', shown:(n,e)=>`当前显示：${n} 节点，${e} 关系`, empty:'当前筛选没有可显示的节点', office:'技术相关官职', highOffice:'高官官职', dynastyName:'朝代', id:'ID', relation:'关系', code:'代码', bureau:'官职类别' },
      en: { title:'CBDB Science and Technical Personnel Network', subtitle:s=>`${s.main} Main Technical Personnel · ${s.high} Related High Officials · ${s.other} Direct People · ${s.bridge} Intermediary People`, language:'Language', dynasty:'Dynasty Filter', allDynasties:'All Dynasties', view:'View Mode', all:'All', direct:'Direct', paths:'High Paths', search:'Search Person', searchPlaceholder:'Name or ID', nodeType:'Node Types', main:'Main Technical Person', high:'High Official', bridge:'Intermediary Person', other:'Direct Person', labels:'Show labels', bureauColor:'Main Office Colors', relationNodes:'Relationship Nodes', directEdge:'Direct Relationship', pathEdge:'High Official Path', shown:(n,e)=>`Showing: ${n} nodes, ${e} relationships`, empty:'No nodes match the current filters', office:'Technical Office', highOffice:'High Office', dynastyName:'Dynasty', id:'ID', relation:'Relation', code:'Code', bureau:'Bureau Type' }
    };
    let lang='zh'; let viewMode='paths'; const enabledTypes=new Set(['main','high','bridge','other']);
    const width=window.innerWidth, height=window.innerHeight;
    const svg=d3.select('#network').append('svg').attr('width',width).attr('height',height);
    const zoomLayer=svg.append('g'), linkLayer=zoomLayer.append('g'), nodeLayer=zoomLayer.append('g'), labelLayer=zoomLayer.append('g');
    const tooltip=d3.select('#tooltip'); const nodeById=new Map(DATA.nodes.map(d=>[d.id,d])); const pathNodeIds=new Set(); DATA.paths.forEach(p=>p.pathIds.forEach(id=>pathNodeIds.add(id))); const bureauByKey=new Map(DATA.bureauGroups.map(g=>[g.key,g]));
    function nodeColor(d){ if(d.nodeType==='main') return (bureauByKey.get(d.bureauType)||DATA.bureauGroups[DATA.bureauGroups.length-1]).color; if(d.nodeType==='high') return 'var(--high)'; if(d.nodeType==='bridge') return 'var(--bridge)'; return 'var(--other)'; }
    const radius=d=>d.nodeType==='main'?7.8:d.nodeType==='high'?7.2:d.nodeType==='bridge'?5.4:3.8;
    const name=d=>lang==='zh'?(d.nameZh||d.nameEn||String(d.id)):(d.nameEn||d.nameZh||String(d.id)); const dynastyName=d=>lang==='zh'?(d.dynastyZh||''):(d.dynastyEn||''); const bureauName=d=>lang==='zh'?(d.bureauZh||''):(d.bureauEn||'');
    const link=linkLayer.selectAll('line').data(DATA.edges).join('line').attr('class',d=>`link ${d.category==='path'?'path':'direct'}`);
    const node=nodeLayer.selectAll('circle').data(DATA.nodes).join('circle').attr('class',d=>`node ${d.nodeType}`).attr('r',radius).style('fill',nodeColor);
    const label=labelLayer.selectAll('text').data(DATA.nodes).join('text').attr('class','label').attr('dy',d=>radius(d)+10).style('font-size',d=>d.nodeType==='main'||d.nodeType==='high'?'9px':'7px').text(name);
    let labelsEnabled=document.getElementById('labels').checked; let framePending=false;
    const simulation=d3.forceSimulation(DATA.nodes).alphaDecay(.05).velocityDecay(.5).force('link',d3.forceLink(DATA.edges).id(d=>d.id).distance(d=>d.category==='path'?42:28).strength(d=>d.category==='path'?.34:.12)).force('charge',d3.forceManyBody().strength(d=>d.nodeType==='main'?-48:d.nodeType==='high'?-42:-13)).force('x',d3.forceX(width*.5).strength(.036)).force('y',d3.forceY(height*.53).strength(.036)).force('center',d3.forceCenter(width/2,height*.53)).force('collide',d3.forceCollide().radius(d=>radius(d)+3)).on('tick',ticked);
    window.setTimeout(()=>simulation.alphaTarget(0).stop(),2800); svg.call(d3.zoom().scaleExtent([.22,4]).on('zoom',e=>zoomLayer.attr('transform',e.transform))); node.call(d3.drag().on('start',dragStart).on('drag',dragged).on('end',dragEnd));
    function ticked(){ if(framePending) return; framePending=true; requestAnimationFrame(()=>{ link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y); node.attr('cx',d=>d.x).attr('cy',d=>d.y); if(labelsEnabled) label.attr('x',d=>d.x).attr('y',d=>d.y); framePending=false; }); }
    function dragStart(event,d){ if(!event.active) simulation.alphaTarget(.22).restart(); d.fx=d.x; d.fy=d.y; } function dragged(event,d){ d.fx=event.x; d.fy=event.y; } function dragEnd(event,d){ if(!event.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }
    function optionText(d){ return lang==='zh'?d.zh:d.en; } function passesDynasty(n,dyn){ return dyn==='ALL'||String(n.dynastyCode)===dyn; } function matchesQuery(n,q){ if(/^\d+$/.test(q)) return String(n.id)===q; return (n.nameZh||'').toLowerCase().includes(q)||(n.nameEn||'').toLowerCase().includes(q); }
    function getVisible(){ const dyn=document.getElementById('dynasty').value; const q=document.getElementById('search').value.trim().toLowerCase(); const visibleIds=new Set(), visibleEdges=new Set(); DATA.edges.forEach((e,idx)=>{ const s=typeof e.source==='object'?e.source.id:e.source; const t=typeof e.target==='object'?e.target.id:e.target; const sn=nodeById.get(s), tn=nodeById.get(t); if(!sn||!tn) return; let keep=enabledTypes.has(sn.nodeType)&&enabledTypes.has(tn.nodeType); if(viewMode==='direct') keep=keep&&e.category==='direct'; if(viewMode==='paths') keep=keep&&e.category==='path'; if(keep&&dyn!=='ALL') keep=passesDynasty(sn,dyn)||passesDynasty(tn,dyn); if(keep&&q) keep=[sn,tn].some(n=>matchesQuery(n,q)); if(keep){ visibleEdges.add(idx); visibleIds.add(s); visibleIds.add(t); } }); if(viewMode==='all'&&!q) DATA.nodes.forEach(n=>{ if(enabledTypes.has(n.nodeType)&&passesDynasty(n,dyn)) visibleIds.add(n.id); }); if(viewMode==='paths'&&!q) DATA.nodes.forEach(n=>{ if(pathNodeIds.has(n.id)&&enabledTypes.has(n.nodeType)&&passesDynasty(n,dyn)) visibleIds.add(n.id); }); if(q) DATA.nodes.forEach(n=>{ if(enabledTypes.has(n.nodeType)&&passesDynasty(n,dyn)&&matchesQuery(n,q)) visibleIds.add(n.id); }); return {visibleIds,visibleEdges}; }
    function applyFilters(){ const v=getVisible(); node.style('display',d=>v.visibleIds.has(d.id)?null:'none'); labelsEnabled=document.getElementById('labels').checked; label.style('display',d=>labelsEnabled&&v.visibleIds.has(d.id)?null:'none').text(name); if(labelsEnabled) label.attr('x',d=>d.x).attr('y',d=>d.y); link.style('display',(d,i)=>v.visibleEdges.has(i)?null:'none'); d3.select('#status').text(i18n[lang].shown(v.visibleIds.size,v.visibleEdges.size)); d3.select('#empty').style('display',v.visibleIds.size?'none':'block').text(i18n[lang].empty); }
    function renderText(){ const t=i18n[lang]; d3.select('#title').text(t.title); d3.select('#subtitle').text(t.subtitle({main:DATA.summary.mainPersonCount, high:DATA.summary.relatedHighOfficialCount, other:DATA.summary.directPersonCount, bridge:DATA.summary.bridgeCount})); d3.select('#languageLabel').text(t.language); d3.select('#dynastyLabel').text(t.dynasty); d3.select('#viewLabel').text(t.view); d3.select('#searchLabel').text(t.search); d3.select('#search').attr('placeholder',t.searchPlaceholder); d3.select('#nodeTypeLabel').text(t.nodeType); d3.select('#toggleMain').text(t.main); d3.select('#toggleHigh').text(t.high); d3.select('#toggleBridge').text(t.bridge); d3.select('#toggleOther').text(t.other); d3.select('#labelsText').text(t.labels); d3.select('#viewAll').text(t.all); d3.select('#viewDirect').text(t.direct); d3.select('#viewPaths').text(t.paths); const dyn=d3.select('#dynasty'); const current=dyn.property('value')||'ALL'; dyn.selectAll('option').remove(); dyn.append('option').attr('value','ALL').text(t.allDynasties); DATA.dynasties.forEach(d=>dyn.append('option').attr('value',d.code).text(optionText(d))); dyn.property('value',current); const bureauLegend=DATA.bureauGroups.map(g=>`<div class="legend-row"><span class="swatch" style="background:${g.color}"></span>${lang==='zh'?g.zh:g.en}</div>`).join(''); d3.select('#legend').html(`<div class="legend-title">${t.bureauColor}</div>${bureauLegend}<div class="legend-title">${t.relationNodes}</div><div class="legend-row"><span class="swatch" style="background:var(--high)"></span>${t.high}</div><div class="legend-row"><span class="swatch" style="background:var(--bridge)"></span>${t.bridge}</div><div class="legend-row"><span class="swatch" style="background:var(--other)"></span>${t.other}</div><div class="legend-row"><span class="swatch-line"></span>${t.directEdge}</div><div class="legend-row"><span class="swatch-line path"></span>${t.pathEdge}</div>`); applyFilters(); }
    d3.select('#language').on('change',function(){ lang=this.value; renderText(); }); d3.select('#dynasty').on('change',applyFilters); d3.select('#search').on('input',applyFilters); d3.select('#labels').on('change',function(){ labelsEnabled=this.checked; applyFilters(); ticked(); }); d3.selectAll('.type-toggle').on('change',function(){ this.checked?enabledTypes.add(this.value):enabledTypes.delete(this.value); applyFilters(); }); d3.selectAll('button.segment').on('click',function(){ viewMode=this.dataset.view; d3.selectAll('button.segment').classed('active',false); d3.select(this).classed('active',true); applyFilters(); });
    node.on('mouseenter',function(event,d){ const t=i18n[lang]; d3.select(this).attr('r',radius(d)+3); const sci=(d.scienceOffices||[]).map(o=>o.zh).slice(0,6).join(' / '); const high=(d.highOffices||[]).map(o=>o.zh).slice(0,6).join(' / '); tooltip.html(`<strong>${name(d)}</strong><div class="muted">${t.id}: ${d.id}</div>${dynastyName(d)?`<div>${t.dynastyName}: ${dynastyName(d)}</div>`:''}${bureauName(d)?`<div>${t.bureau}: ${bureauName(d)}</div>`:''}${sci?`<div>${t.office}: ${sci}</div>`:''}${high?`<div>${t.highOffice}: ${high}</div>`:''}`).style('opacity',1).style('left',`${event.clientX+14}px`).style('top',`${event.clientY+14}px`); }).on('mousemove',event=>tooltip.style('left',`${event.clientX+14}px`).style('top',`${event.clientY+14}px`)).on('mouseleave',function(event,d){ d3.select(this).attr('r',radius(d)); tooltip.style('opacity',0); });
    link.on('mouseenter',function(event,d){ const t=i18n[lang]; d3.select(this).attr('stroke-width',d.category==='path'?3:2.4).attr('stroke-opacity',1); tooltip.html(`<strong>${d.category==='path'?t.pathEdge:t.directEdge}</strong><div>${t.relation}: ${d.type||d.kind}</div><div class="muted">${t.code}: ${d.code}</div>`).style('opacity',1).style('left',`${event.clientX+14}px`).style('top',`${event.clientY+14}px`); }).on('mousemove',event=>tooltip.style('left',`${event.clientX+14}px`).style('top',`${event.clientY+14}px`)).on('mouseleave',function(event,d){ d3.select(this).attr('stroke-width',d.category==='path'?1.45:1.1).attr('stroke-opacity',null); tooltip.style('opacity',0); });
    renderText();
  </script>
</body>
</html>
'''
    return template.replace("__DATA_JSON__", data_json).replace("__D3_SOURCE__", d3_source)


def main() -> None:
    data, review = build_data()
    errors = validate_data(data)
    if errors:
        raise RuntimeError("Data validation failed: " + "; ".join(errors[:10]))
    JSON_OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    HTML_OUT.write_text(render_html(data))

    print("STEP REVIEW 1 - project/script")
    print(f"  script: {Path(__file__).resolve()}")
    print(f"  db: {DB_PATH}")
    print(f"  outputs: {JSON_OUT.name}, {HTML_OUT.name}")
    print("STEP REVIEW 2 - technical extraction")
    print(f"  all technical offices: {data['summary']['scienceOfficeCount']}")
    print(f"  used technical offices: {data['summary']['usedScienceOfficeCount']}")
    print(f"  main people: {data['summary']['mainPersonCount']}")
    print(f"  bureau counts: {data['summary']['bureauTypeCounts']}")
    print("  technical samples:")
    for key, rows in review["technicalOfficeSamples"].items():
        if rows:
            print(f"    {key}: " + "; ".join(f"{r['id']}:{r['zh']}" for r in rows[:4]))
    print("STEP REVIEW 3/4 - high officials")
    print(f"  high official candidates: {data['summary']['highOfficialCandidateCount']}")
    print(f"  related high officials in network: {data['summary']['relatedHighOfficialCount']}")
    print("  top high offices: " + "; ".join(f"{name}({count})" for name, count in data['summary']['topHighOfficialOffices'][:12]))
    print("STEP REVIEW 5/6 - network")
    print(f"  direct people: {data['summary']['directPersonCount']}")
    print(f"  bridge people: {data['summary']['bridgeCount']}")
    print(f"  nodes: {data['summary']['nodeCount']}, edges: {data['summary']['edgeCount']}, paths: {data['summary']['pathCount']}")
    print(f"  main with direct relations: {data['summary']['mainWithDirectRelationsCount']}")
    print(f"  main with high paths: {data['summary']['highPathCoveredMainCount']}")
    print("STEP REVIEW 7/8 - validation")
    print("  JSON parsed and validated: OK")
    print("  HTML generated with embedded D3 and data: OK")


if __name__ == "__main__":
    main()
