# -*- coding: utf-8 -*-
import time
import json
import csv
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any


def _now_s():
    return time.time()


class LatencyTracer:
    """
    🔧 改良版 LatencyTracer（v2.2）

    改善点:
    - frame/tid/n_det などメタ情報を自動でCSV先頭列に
    - 未定義のmetaキーもJSONLには保存（柔軟性UP）
    - オプションで標準出力ログを追加（debug=True）
    - flush()が自動でフォルダを作成
    """

    CSV_COLUMNS = [
        "frame",       # 実行フレーム番号
        "stage",       # 処理モジュール名
        "latency_ms",  # 実行時間[ms]
        "gpu",         # GPU同期したか
        "tid",         # トラックIDなど
        "n_det",       # 検出数
        "hist_len",    # 履歴長
        "warmup",      # ウォームアップ期間中か
        "error",       # 例外情報
        "ts_start",    # 開始時刻（秒）
        "ts_end"       # 終了時刻（秒）
    ]

    def __init__(
        self,
        csv_path: str = "risa_latency.csv",
        jsonl_path: str = "risa_latency.jsonl",
        buffer_size: int = 100,
        debug: bool = False
    ):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.buffer_size = buffer_size
        self.debug = debug
        self._csv_buffer = []
        self._jsonl_buffer = []

        # 出力フォルダを自動生成
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        self._init_csv()

        # torchが利用可能ならGPU同期も対応
        try:
            import torch
            self._torch = torch
        except Exception:
            self._torch = None

    def _init_csv(self):
        """CSVが存在しない場合にヘッダー行を書き込む"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()

    def _gpu_sync(self):
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    @contextmanager
    def span(self, name: str, meta: Optional[Dict[str, Any]] = None, gpu: bool = False):
        """
        処理モジュール単位でレイテンシを計測するコンテキストマネージャ

        Args:
            name: 処理モジュール名（例: "yolo_detect"）
            meta: フレーム番号などのメタデータ（例: {"frame": 123, "tid": 2}）
            gpu: TrueならGPU同期を計測境界に入れる
        """
        if gpu:
            self._gpu_sync()
        t0 = _now_s()
        err = None

        try:
            yield
        except Exception as e:
            err = repr(e)
            raise
        finally:
            if gpu:
                self._gpu_sync()
            t1 = _now_s()
            dur_ms = round((t1 - t0) * 1000, 3)

            # === CSV行 ===
            row = {col: None for col in self.CSV_COLUMNS}
            row.update({
                "stage": name,
                "latency_ms": dur_ms,
                "gpu": gpu,
                "error": err,
                "ts_start": t0,
                "ts_end": t1,
            })

            # meta情報を反映（frameなど）
            if isinstance(meta, dict):
                for k, v in meta.items():
                    if k in row:
                        row[k] = v

            self._csv_buffer.append(row)

            # === JSONL ===
            json_row = dict(row)
            if isinstance(meta, dict):
                # 未定義metaも保持
                for k, v in meta.items():
                    if k not in json_row:
                        json_row[k] = v
            self._jsonl_buffer.append(json_row)

            if self.debug:
                frame_str = f"[frame={meta.get('frame', '?')}]" if meta else ""
                print(f"⏱️ {frame_str} {name}: {dur_ms:.2f} ms")

            if len(self._csv_buffer) >= self.buffer_size:
                self.flush()

    def flush(self):
        """CSVとJSONLにバッファを書き出す"""
        if not self._csv_buffer:
            return

        # CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writerows(self._csv_buffer)

        # JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            for row in self._jsonl_buffer:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self._csv_buffer.clear()
        self._jsonl_buffer.clear()

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass


# ========== デモ実行例 ==========
if __name__ == "__main__":
    import numpy as np
    tr = LatencyTracer("test_latency.csv", "test_latency.jsonl", debug=True)

    for frame_idx in range(5):
        with tr.span("detect_yolo", meta={"frame": frame_idx, "warmup": frame_idx < 2}, gpu=True):
            time.sleep(0.015)
        with tr.span("track_bytetrack", meta={"frame": frame_idx}, gpu=False):
            time.sleep(0.005)
        with tr.span("llava_eval", meta={"frame": frame_idx, "tid": 1}, gpu=True):
            time.sleep(0.020)

    tr.flush()
    print("✅ 計測完了: test_latency.csv / test_latency.jsonl")
