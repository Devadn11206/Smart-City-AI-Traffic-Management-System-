import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
# data paths under the smart_city_ai package root
CSV = ROOT / 'results' / 'emergency_metrics' / 'stop_audit_short' / 'stop_audit_debug.csv'
if not CSV.exists():
    # fallback to earlier stop_audit folder
    CSV = ROOT / 'results' / 'emergency_metrics' / 'stop_audit' / 'stop_audit_debug.csv'
OUT = ROOT / 'results' / 'response_analysis' / 'emergency_aco_coordination_report.json'
OUT.parent.mkdir(parents=True, exist_ok=True)

# read CSV rows
rows = []
with CSV.open('r', encoding='utf-8') as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        # skip duplicated header rows that may appear in rolling logs
        if (r.get('step') or '').strip().lower() == 'step':
            continue
        # normalize fields
        try:
            r['step'] = int(r.get('step') or 0)
        except Exception:
            continue
        try:
            r['current_speed'] = float(r.get('current_speed') or 0.0)
        except Exception:
            r['current_speed'] = 0.0
        # parse cleared/rerouted id lists
        for k in ('vehicles_cleared_ids','vehicles_rerouted_ids'):
            v = (r.get(k) or '').strip()
            r[k] = [s for s in v.split(';') if s]
        rows.append(r)

# organize by vehicle
by_vid = defaultdict(list)
for r in rows:
    by_vid[r['vehicle_id']].append(r)

report = {
    'per_vehicle': {},
    'summary': {},
}

# helper to detect stops
all_stops = []
for vid, seq in by_vid.items():
    seq.sort(key=lambda x: x['step'])
    prev_speed = None
    current_stop = None
    stops = []
    for row in seq:
        speed = row['current_speed']
        if prev_speed is None:
            prev_speed = speed
        # start stop
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
        # end stop
        if prev_speed <= 0.1 and speed > 0.1 and current_stop is not None:
            current_stop['end_step'] = row['step']
            current_stop['duration_steps'] = current_stop['end_step'] - current_stop['start_step']
            # decide reason heuristically
            ctx = current_stop['start_context']
            reason = 'unknown'
            if ctx['vehicles_cleared_ids']:
                reason = 'vehicle_blocking'
            elif ctx['signalized_intersections_on_route'] and ctx['predictive_corridor_tls'] == 0:
                reason = 'red_light'
            elif ctx['vehicles_rerouted_ids'] and ctx['vehicles_cleared_ids']==[]:
                reason = 'aco_reroute_attempted_but_blocked'
            elif ctx['active_corridor_intersections'] == 0 and ctx['predictive_corridor_tls']>0:
                reason = 'corridor_activated_too_late'
            current_stop['reason'] = reason
            stops.append(current_stop)
            all_stops.append(current_stop)
            current_stop = None
        prev_speed = speed
    # attach stops
    report['per_vehicle'][vid] = {
        'stops': stops,
        'stop_count': len(stops),
        'total_waiting_steps': sum(s['duration_steps'] for s in stops)
    }

# compute summary metrics
total_stops = len(all_stops)
avg_duration = float(sum(s['duration_steps'] for s in all_stops) / total_stops) if total_stops else 0.0
counts = defaultdict(int)
for s in all_stops:
    counts[s['reason']] += 1

report['summary'] = {
    'total_stop_count': total_stops,
    'average_stop_duration_steps': avg_duration,
    'counts_by_reason': dict(counts),
}

# dominant cause
dominant = 'unknown'
if counts:
    dominant = max(counts.items(), key=lambda x: x[1])[0]
report['dominant_cause'] = dominant

# write out
OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Wrote', OUT)
