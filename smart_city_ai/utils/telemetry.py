from __future__ import annotations

import csv
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class TelemetryRecord:
    timestamp: float
    scenario_id: str
    run_id: str
    source: str
    step: int | None = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        row = {
            'timestamp': self.timestamp,
            'scenario_id': self.scenario_id,
            'run_id': self.run_id,
            'source': self.source,
            'step': self.step,
        }
        row.update(self.payload)
        return row


class TelemetryWriter:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.jsonl_dir = self.root_dir / 'jsonl'
        self.csv_dir = self.root_dir / 'csv'
        self.sqlite_path = self.root_dir / 'telemetry.sqlite'
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_sqlite()

    def _ensure_sqlite(self) -> None:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS telemetry (
                    timestamp REAL,
                    scenario_id TEXT,
                    run_id TEXT,
                    source TEXT,
                    step INTEGER,
                    payload_json TEXT
                )
                '''
            )
            conn.commit()
        finally:
            conn.close()

    def _jsonl_path(self, scenario_id: str, run_id: str) -> Path:
        return self.jsonl_dir / f'{scenario_id}__{run_id}.jsonl'

    def _csv_path(self, scenario_id: str, run_id: str) -> Path:
        return self.csv_dir / f'{scenario_id}__{run_id}.csv'

    def write(self, record: TelemetryRecord) -> None:
        row = record.to_row()
        jsonl_path = self._jsonl_path(record.scenario_id, record.run_id)
        csv_path = self._csv_path(record.scenario_id, record.run_id)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with jsonl_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, sort_keys=True) + '\n')

        header = list(row.keys())
        write_header = not csv_path.exists()
        with csv_path.open('a', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute(
                'INSERT INTO telemetry(timestamp, scenario_id, run_id, source, step, payload_json) VALUES (?, ?, ?, ?, ?, ?)',
                (
                    float(record.timestamp),
                    record.scenario_id,
                    record.run_id,
                    record.source,
                    int(record.step) if record.step is not None else None,
                    json.dumps(record.payload, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()


class TelemetryCollector:
    def __init__(self, root_dir: str | Path, scenario_id: str, run_id: str):
        self.writer = TelemetryWriter(root_dir)
        self.scenario_id = scenario_id
        self.run_id = run_id

    def emit(self, source: str, step: int | None = None, **payload: Any) -> TelemetryRecord:
        record = TelemetryRecord(
            timestamp=time.time(),
            scenario_id=self.scenario_id,
            run_id=self.run_id,
            source=source,
            step=step,
            payload=payload,
        )
        self.writer.write(record)
        return record


def summarize_telemetry(root_dir: str | Path) -> Dict[str, Any]:
    root = Path(root_dir)
    sqlite_path = root / 'telemetry.sqlite'
    summary: Dict[str, Any] = {'root_dir': str(root), 'exists': sqlite_path.exists(), 'runs': 0, 'records': 0, 'sources': {}}
    if not sqlite_path.exists():
        return summary
    conn = sqlite3.connect(sqlite_path)
    try:
        rows = conn.execute('SELECT scenario_id, run_id, source, COUNT(*) FROM telemetry GROUP BY scenario_id, run_id, source').fetchall()
        summary['records'] = int(conn.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0])
        summary['runs'] = int(conn.execute('SELECT COUNT(DISTINCT scenario_id || "::" || run_id) FROM telemetry').fetchone()[0])
        for scenario_id, run_id, source, count in rows:
            summary['sources'].setdefault(source, 0)
            summary['sources'][source] += int(count)
    finally:
        conn.close()
    return summary
