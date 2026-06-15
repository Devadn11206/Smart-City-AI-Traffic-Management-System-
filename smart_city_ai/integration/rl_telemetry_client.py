import json
import urllib.request
import urllib.error


def send_telemetry(payload: dict, host: str = '127.0.0.1', port: int = 8765, timeout: float = 2.0) -> bool:
    url = f'http://{host}:{port}/telemetry'
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except urllib.error.URLError:
        return False
    except Exception:
        return False


class RLTelemetryPublisher:
    def __init__(self, host: str = '127.0.0.1', port: int = 8765):
        self.host = host
        self.port = port

    def publish(self, *, step=None, algorithm=None, last_reward=None, avg_queue=None, avg_wait=None, decision_time=None, actions=None, agent=None):
        payload = {
            'step': step,
            'algorithm': algorithm,
            'last_reward': last_reward,
            'avg_queue': avg_queue,
            'avg_wait': avg_wait,
            'decision_time': decision_time,
            'actions': actions,
            'agent': agent,
            'timestamp': __import__('time').time(),
        }
        return send_telemetry(payload, host=self.host, port=self.port)


if __name__ == '__main__':
    # quick smoke test
    ok = send_telemetry({'step': 1, 'algorithm': 'ddqn', 'last_reward': 0.0})
    print('sent:', ok)
