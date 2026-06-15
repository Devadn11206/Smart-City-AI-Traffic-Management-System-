import threading
import time
import json
from pathlib import Path

_monitor_thread = None


def start_monitoring(out_dir: str = 'results/monitoring', interval: float = 10.0, pid: int | None = None):
    """Start a background thread to record CPU and memory stats periodically.

    Writes a JSONL file with timestamp, rss, vms, cpu_percent, and process_cpu_seconds.
    """
    global _monitor_thread
    try:
        import psutil
    except Exception:
        # psutil is optional for monitoring; if unavailable, skip starting monitor
        return None
    if _monitor_thread is not None and _monitor_thread.is_alive():
        return
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logfile = out_path / f'monitor_{int(time.time())}.jsonl'
    proc = psutil.Process(pid) if pid else psutil.Process()

    def _run():
        with logfile.open('a', encoding='utf-8') as fh:
            while True:
                try:
                    ts = time.time()
                    mem = proc.memory_info()
                    cpu = proc.cpu_percent(interval=None)
                    entry = {
                        'timestamp': ts,
                        'rss': mem.rss,
                        'vms': mem.vms,
                        'cpu_percent': cpu,
                        'cpu_times': proc.cpu_times()._asdict(),
                    }
                    fh.write(json.dumps(entry) + '\n')
                    fh.flush()
                except Exception:
                    try:
                        fh.write(json.dumps({'timestamp': time.time(), 'error': 'monitoring exception'}) + '\n')
                        fh.flush()
                    except Exception:
                        pass
                time.sleep(interval)

    _monitor_thread = threading.Thread(target=_run, daemon=True)
    _monitor_thread.start()
    return logfile
