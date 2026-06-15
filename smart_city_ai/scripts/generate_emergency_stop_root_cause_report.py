import csv
import json
from pathlib import Path
from collections import defaultdict


def parse_stop_csv(csv_path: Path):
    if not csv_path.exists():
        return None
    rows = []
    with csv_path.open('r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # skip duplicated header rows
            if (r.get('step') or '').strip().lower() == 'step':
                continue
            try:
                r['step'] = int(r.get('step') or 0)
            except Exception:
                continue
            try:
                r['current_speed'] = float(r.get('current_speed') or 0.0)
            except Exception:
                r['current_speed'] = 0.0
            for k in ('vehicles_cleared_ids','vehicles_rerouted_ids'):
                v = (r.get(k) or '').strip()
                r[k] = [s for s in v.split(';') if s]
            rows.append(r)

    by_vid = defaultdict(list)
    for r in rows:
        by_vid[r['vehicle_id']].append(r)

    all_stops = []
    per_vehicle = {}

    for vid, seq in by_vid.items():
        seq.sort(key=lambda x: x['step'])
        prev_speed = None
        current_stop = None
        stops = []
        for row in seq:
            speed = row['current_speed']
            if prev_speed is None:
                prev_speed = speed
            if prev_speed > 0.1 and speed <= 0.1 and current_stop is None:
                current_stop = {
                    'vehicle_id': vid,
                    'start_step': row['step'],
                    'edge': row.get('current_edge'),
                    'speed_before_stop': prev_speed,
                    'start_context': {
                        'active_corridor_intersections': int(row.get('active_corridor_intersections') or 0),
                        'predictive_corridor_tls': int(row.get('predictive_corridor_tls') or 0),
                        'signalized_intersections_on_route': int(row.get('signalized_intersections_on_route') or 0),
                        'vehicles_cleared_ids': row.get('vehicles_cleared_ids') or [],
                        'vehicles_rerouted_ids': row.get('vehicles_rerouted_ids') or [],
                    }
                }
            if prev_speed <= 0.1 and speed > 0.1 and current_stop is not None:
                current_stop['end_step'] = row['step']
                current_stop['duration_steps'] = current_stop['end_step'] - current_stop['start_step']
                ctx = current_stop['start_context']
                reason = 'unknown'
                if ctx['vehicles_cleared_ids']:
                    reason = 'vehicle_blocking'
                elif ctx['signalized_intersections_on_route'] and ctx['predictive_corridor_tls'] == 0:
                    reason = 'red_light'
                elif ctx['vehicles_rerouted_ids'] and not ctx['vehicles_cleared_ids']:
                    reason = 'aco_reroute_attempted_but_blocked'
                elif ctx['active_corridor_intersections'] == 0 and ctx['predictive_corridor_tls'] > 0:
                    reason = 'corridor_activated_too_late'
                current_stop['reason'] = reason
                stops.append(current_stop)
                all_stops.append(current_stop)
                current_stop = None
            prev_speed = speed

        per_vehicle[vid] = {
            'stops': stops,
            'stop_count': len(stops),
            'total_waiting_steps': sum(s['duration_steps'] for s in stops)
        }

    total_stops = len(all_stops)
    avg_duration = float(sum(s['duration_steps'] for s in all_stops) / total_stops) if total_stops else 0.0
    counts = defaultdict(int)
    for s in all_stops:
        counts[s['reason']] += 1

    dominant = 'unknown'
    if counts:
        dominant = max(counts.items(), key=lambda x: x[1])[0]

    return {
        'per_vehicle': per_vehicle,
        'summary': {
            'total_stop_count': total_stops,
            'average_stop_duration_steps': avg_duration,
            'counts_by_reason': dict(counts),
        },
        'dominant_cause': dominant,
    }


def compare_reports(before, after):
    if before is None or after is None:
        return {'note': 'one or both reports missing', 'before_exists': before is not None, 'after_exists': after is not None}
    b = before['summary']
    a = after['summary']
    comp = {
        'before_total_stops': b.get('total_stop_count', 0),
        'after_total_stops': a.get('total_stop_count', 0),
        'delta_total_stops': a.get('total_stop_count', 0) - b.get('total_stop_count', 0),
        'before_avg_duration_steps': b.get('average_stop_duration_steps', 0.0),
        'after_avg_duration_steps': a.get('average_stop_duration_steps', 0.0),
        'delta_avg_duration_steps': a.get('average_stop_duration_steps', 0.0) - b.get('average_stop_duration_steps', 0.0),
        'counts_by_reason': {}
    }
    reasons = set(list(b.get('counts_by_reason', {}).keys()) + list(a.get('counts_by_reason', {}).keys()))
    for r in reasons:
        comp['counts_by_reason'][r] = {
            'before': b.get('counts_by_reason', {}).get(r, 0),
            'after': a.get('counts_by_reason', {}).get(r, 0),
            'delta': a.get('counts_by_reason', {}).get(r, 0) - b.get('counts_by_reason', {}).get(r, 0)
        }
    return comp


def main():
    ROOT = Path(__file__).resolve().parent.parent
    pre_csv = ROOT / 'results' / 'emergency_metrics' / 'stop_audit' / 'stop_audit_debug.csv'
    post_csv = ROOT / 'results' / 'emergency_metrics' / 'stop_audit_after' / 'stop_audit_debug.csv'
    if not pre_csv.exists():
        short = ROOT / 'results' / 'emergency_metrics' / 'stop_audit_short' / 'stop_audit_debug.csv'
        if short.exists():
            pre_csv = short

    before = parse_stop_csv(pre_csv) if pre_csv.exists() else None
    after = parse_stop_csv(post_csv) if post_csv.exists() else None

    report = {
        'before': before,
        'after': after,
        'comparison': compare_reports(before, after)
    }

    out = ROOT / 'results' / 'response_analysis' / 'emergency_stop_root_cause_report.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote', out)


if __name__ == '__main__':
    main()
