# NVIDIA Blueprint 橋接

Exporter 依 [官方 AST discovery](https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint/blob/main/core/discovery.py) 產生一個實驗一個檔案、一個 typed public function、`-> dict`。不要求 Blueprint 理解 class/decorator。

Wrapper 呼叫 QICK_WORKER_URL；worker 使用 Session，與日常 Notebook 的 Measurement 共用 native backend、資料、分析工具。Result 含 tagged scalar/array、base64 PNG、run ID、quality、backend 標記。適量 I/Q/axes **實際匯入** Blueprint；完整 shots 留在 SDK RunStore。

已用 NVIDIA 官方 discovery、runner、storage、models 驗證 script discovery、validation、JSON/array parsing、HDF5/SQLite 和 PNG roundtrip。SHA256 與结果：[verification/blueprint.json](verification/blueprint.json)。未啟動 LLM provider 或透過 agent 操作實體板卡。

## 設定

```powershell
.venv\Scripts\qickworkspace export-blueprint lab/blueprint_scripts
.venv\Scripts\qickworkspace serve
$env:QICK_WORKER_URL = "http://127.0.0.1:8000"
```

將 wrappers 放到 Blueprint 的 scripts_dir，或把 scripts_dir 指向生成位置。Blueprint 使用的 Python 需能 import SDK 和 httpx；它不持有 QICK 連線。

同一 logical measurement 的重試沿用 `request_id`；不同參數重用 ID 會回 409。省略 ID 代表新量測，不會把正常的重複量測合併。

## Worker API

| 路徑 | 行為 |
|---|---|
| `GET /health` / `GET /device` | 狀態、device、capabilities |
| `GET /experiments` | ID、version、完整 parameter schema |
| `POST /experiments/check` | schema/resource/compile |
| `POST /experiments/run` | request_id 去重、排隊 |
| `GET /experiments/{id}` | queued/running/completed/failed/cancelled/interrupted |
| `GET /experiments/{id}/result` | Blueprint-compatible result |
| `POST /experiments/{id}/cancel` | cooperative cancel |
| `GET /runs` | durable run journal |

Wrappers 等待最多 240 秒，低於 Blueprint 常見 300 秒 subprocess timeout。更長量測用 WorkerClient submit/status/result/cancel；timeout 後 worker 可能仍執行，error 保留 job ID，不會自動重啟量測。

Agent 執行 wrapper 不會自動提交 calibration。原始 Pydantic schema 在 worker 執行，補足 Blueprint 對 enum/複雜型別的限制。

預設 API 綁 localhost，未內建多租戶帳號與 TLS。遠端部署請接到實驗室既有 proxy、認證與網路管理；本機 SDK 可用不代表雲端產品已認證。
