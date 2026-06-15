from http.server import BaseHTTPRequestHandler, HTTPServer
import argparse
import json
import time
from pathlib import Path
import threading
import csv
import logging

logger = logging.getLogger(__name__)


class TelemetryProcessor:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.telemetry_dir = self.results_dir / 'rl' / 'telemetry'
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.timeseries_path = self.results_dir / 'rl' / 'telemetry_timeseries.csv'
        self.lock = threading.Lock()

    def process(self, payload: dict) -> None:
        ts = int(time.time())
        # Write latest JSON snapshot (timestamped with _latest.json suffix to match dashboard reader)
        latest_path = self.telemetry_dir / f"{ts}_latest.json"
        try:
            with self.lock:
                latest_path.write_text(json.dumps(payload, indent=2))
        except Exception:
            logger.exception('Failed to write telemetry latest JSON to %s', latest_path)

        # Append a timeseries row for quick plotting
        row = {
            'step': payload.get('step'),
            'timestamp': ts,
            'algorithm': payload.get('algorithm'),
            'last_reward': payload.get('last_reward'),
            'avg_queue': payload.get('avg_queue'),
            'avg_wait': payload.get('avg_wait'),
            'decision_time': payload.get('decision_time'),
            'agent': payload.get('agent', ''),
        }

        header = ['step', 'timestamp', 'algorithm', 'last_reward', 'avg_queue', 'avg_wait', 'decision_time', 'agent']
        try:
            with self.lock:
                first = not self.timeseries_path.exists()
                with open(self.timeseries_path, 'a', newline='') as fh:
                    writer = csv.DictWriter(fh, fieldnames=header)
                    if first:
                        writer.writeheader()
                    writer.writerow(row)
        except Exception:
            logger.exception('Failed to append telemetry timeseries row to %s', self.timeseries_path)


class TelemetryHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Accept JSON POSTs to any path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else b''
        try:
            payload = json.loads(body.decode('utf-8')) if body else {}
        except Exception:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'invalid json')
            return

        # Process payload using server.processor
        try:
            self.server.processor.process(payload)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
        except Exception:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'error')

    def log_message(self, format, *args):
        # keep server quiet by default
        return


def run_server(host: str, port: int, results_dir: Path):
    processor = TelemetryProcessor(results_dir)
    server = HTTPServer((host, port), TelemetryHandler)
    server.processor = processor
    print(f'RL telemetry server listening on http://{host}:{port} -> writing to {processor.telemetry_dir}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('shutting down')
        server.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Telemetry HTTP Server')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8765, type=int)
    parser.add_argument('--results', default=str(Path(__file__).resolve().parent.parent / 'results'))
    args = parser.parse_args()
    run_server(args.host, args.port, Path(args.results))
