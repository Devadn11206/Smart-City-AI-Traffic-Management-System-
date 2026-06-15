import os
import sys
import json
from collections import defaultdict

# ensure smart_city_ai is on path so local packages import like other scripts do
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
smart_city_ai_path = os.path.join(repo_root, 'smart_city_ai')
sys.path.insert(0, smart_city_ai_path)

from sumolib import net
from aco_routing.pheromone_manager import PheromoneManager
from aco_routing.aco_router import ACORouter


def build_adjacency_and_edgeinfo_from_net(netfile):
    adjacency = {}
    edge_info = {}
    netobj = net.readNet(netfile)
    for edge in netobj.getEdges():
        eid = edge.getID()
        adjacency.setdefault(eid, [])
        edge_info[eid] = {'length': edge.getLength()}
        for out in edge.getToNode().getOutgoing():
            adjacency[eid].append(out.getID())
    return adjacency, edge_info, netobj


def sample_pairs_from_debug_csv(csv_path, max_pairs=5):
    pairs = []
    seen = set()
    if not os.path.exists(csv_path):
        return pairs
    with open(csv_path, 'r', encoding='utf-8') as fh:
        # skip header
        hdr = fh.readline()
        for line in fh:
            if not line.strip():
                continue
            parts = line.split(',')
            # current_route is second column
            if len(parts) < 2:
                continue
            route = parts[1].strip()
            if '>' in route:
                edges = [e for e in route.split('>') if e]
                if len(edges) >= 2:
                    src = edges[0]
                    tgt = edges[-1]
                    key = (src, tgt)
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
                        if len(pairs) >= max_pairs:
                            break
    return pairs


def probe_pair(src, tgt, adjacency, edge_info, pheromone, netobj, aco, edge_ids):
    report = {}
    report['source'] = src
    report['target'] = tgt
    report['source_internal'] = src.startswith(':') if isinstance(src, str) else False
    report['target_internal'] = tgt.startswith(':') if isinstance(tgt, str) else False

    # adjacency[source]
    adj_src = adjacency.get(src, None)
    report['adjacency_source'] = adj_src if adj_src is not None else []
    report['num_neighbors'] = len(adj_src) if adj_src is not None else 0

    # pheromone entries for source
    pher_entries = []
    for (u, v), val in pheromone.pheromones.items():
        if u == src:
            pher_entries.append({'to': v, 'value': float(val)})
    report['pheromone_entries_from_source'] = pher_entries

    # heuristic values for neighbors
    heuristics = {}
    for v in (adj_src or []):
        heuristics[v] = float(edge_info.get(v, {}).get('length', 1.0))
    report['heuristic_by_neighbor'] = heuristics

    # verify identifier consistency: check net has edge and lanes
    report['source_in_net'] = (netobj.getEdge(src) is not None) if src in edge_ids else False
    report['target_in_net'] = (netobj.getEdge(tgt) is not None) if tgt in edge_ids else False

    # lane ids for source and target
    try:
        src_edge = netobj.getEdge(src) if src in edge_ids else None
        tgt_edge = netobj.getEdge(tgt) if tgt in edge_ids else None
        report['source_lanes'] = [lane.getID() for lane in src_edge.getLanes()] if src_edge is not None else []
        report['target_lanes'] = [lane.getID() for lane in tgt_edge.getLanes()] if tgt_edge is not None else []
    except Exception:
        report['source_lanes'] = []
        report['target_lanes'] = []

    # adjacency keys sample and pheromone keys sample
    report['adjacency_keys_sample'] = list(list(adjacency.keys())[:50])
    report['pheromone_keys_sample'] = [f"{u}->{v}" for (u, v) in list(pheromone.pheromones.keys())[:200]]

    # mismatch booleans
    report['source_key_exists'] = src in adjacency
    report['target_key_exists'] = tgt in adjacency
    report['neighbors_found'] = bool(adj_src)
    report['pheromone_entry_found'] = len(pher_entries) > 0

    # attempts: run ACORouter.find_route
    try:
        path, cost = aco.find_route(src, tgt, adjacency)
        report['aco_path'] = path
        report['aco_cost'] = float(cost) if cost is not None else None
        aco_debug = getattr(aco, 'last_debug_info', {}) or {}
        report['aco_debug'] = aco_debug
    except Exception as e:
        report['aco_exception'] = str(e)

    # If adjacency[source] empty: try to inspect net outgoing
    if not adj_src:
        reason = {}
        if src in edge_ids:
            out_edges = [o.getID() for o in netobj.getEdge(src).getToNode().getOutgoing()]
            reason['net_outgoing_count'] = len(out_edges)
            reason['net_outgoing_sample'] = out_edges[:20]
        else:
            reason['net_has_edge'] = False
        report['adjacency_empty_diagnosis'] = reason

    # identify exact key mismatches if any (e.g., presence of ':' prefixes in net but not adjacency)
    key_mismatch = {}
    # check for alternative keys with/without ':'
    if not report['source_key_exists'] and isinstance(src, str) and src.startswith(':'):
        alt = src.lstrip(':')
        key_mismatch['alt_source_stripped_colon'] = alt in adjacency
    if not report['target_key_exists'] and isinstance(tgt, str) and tgt.startswith(':'):
        alt = tgt.lstrip(':')
        key_mismatch['alt_target_stripped_colon'] = alt in adjacency
    report['key_mismatch_checks'] = key_mismatch

    return report


def classify_reports(pair_reports):
    classes = set()
    for r in pair_reports:
        if not r['source_key_exists'] or not r['target_key_exists']:
            # if either missing, likely adjacency key mismatch or internal edge issue
            if r['source_internal'] or r['target_internal']:
                classes.add('Internal SUMO Edge Issue')
            else:
                classes.add('Adjacency Key Mismatch')
        if r['source_key_exists'] and not r['neighbors_found']:
            classes.add('Graph Connectivity Issue')
        if not r['pheromone_entry_found']:
            classes.add('Pheromone Table Issue')
        # heuristic check: if heuristics for neighbors empty while neighbors exist
        if r['neighbors_found'] and not r['heuristic_by_neighbor']:
            classes.add('Heuristic Issue')
    if not classes:
        classes.add('Other')
    return sorted(list(classes))


def main():
    netfile = os.path.join('smart_city_ai', 'sumo_network', 'network.net.xml')
    adjacency, edge_info, netobj = build_adjacency_and_edgeinfo_from_net(netfile)
    # precompute edge id set for compatibility with sumolib Net API
    edge_ids = set(e.getID() for e in netobj.getEdges())
    pheromone = PheromoneManager(adjacency, initial_pheromone=1.0, evaporation_rate=0.03)
    aco = ACORouter(pheromone, lambda e: edge_info.get(e, {}).get('length', 1.0), alpha=1.0, beta=2.0, ants=30)

    csv_path = os.path.join('results', 'aco_debug_log.csv')
    pairs = sample_pairs_from_debug_csv(csv_path, max_pairs=10)
    if not pairs:
        print('No pairs sampled from', csv_path)
        return 1
    # limit to 5
    pairs = pairs[:5]

    reports = []
    for src, tgt in pairs:
        print(f'Probing {src} -> {tgt}')
        r = probe_pair(src, tgt, adjacency, edge_info, pheromone, netobj, aco, edge_ids)
        reports.append(r)

    summary = {
        'pairs_tested': len(reports),
        'pairs': reports,
        'classification': classify_reports(reports),
    }

    out_path = os.path.join('results', 'aco_key_consistency_report.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print('Wrote', out_path)


if __name__ == '__main__':
    sys.exit(main())
