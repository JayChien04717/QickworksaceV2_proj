# QickworkspaceV2：目前架構與日常操作

本文件對應目前程式結構。日常操作細節見 [Notebook API](NOTEBOOK_API.md)，
檔案儲存見 [Labber 文件](LABBER.md) 與 [labtools](../labtools/README.md)。

## 目錄與責任

目前 repository 外層仍是 `QickworksaceV2/`，Python 套件名是
`QickworkspaceV2`。外層改名為 `QickworkspaceV2_proj/` 尚未完成；現有指令
仍使用實際路徑，不能只改資料夾名稱而不更新啟動路徑及安裝環境。

```text
QickworksaceV2/
  ChipCalibration.ipynb          日常校正入口
  QickworkspaceV2.ipynb          直接 Measurement 使用方式
  notebooks/advanced_measurements.ipynb
  lab/                          專案、接線 profile 與工作設定
  QickworkspaceV2/
    experiments/<experiment>/
      program.py                原生 Program 與 build(ctx, parameters)
      parameters.py             catalog 參數 schema、預設值與範圍
      analysis.py               實驗分析、最後繪圖、校正更新規則
      __init__.py               ExperimentSpec 與公開 imports
    analysis/traces.py          選擇 SDK trace 並呼叫數值 fitter
    data/models.py              ExperimentData、TraceData
    data/store.py               執行紀錄與分析版本
    data/transport.py           worker 結果與 artifact 契約
    runtime/                    Measurement、Session、catalog、worker
    notebook.py                 NotebookLab 執行、展示與工作校正
  labtools/
    fitting/functions.py        公式與各模型 fit 函式
    fitting/solver.py           接受 callable 的數值求解器
    fitting/result.py           FitResult
    fitting/fit_n_res.py        複數 IQ 寬頻共振器偵測
    hdf5/                      通用 HDF5 讀寫及原子替換
    labber/                    Labber 格式與匯出
    catalog.py                 SQLite 檔案索引、查詢及重建
    serialization.py           共用 JSON metadata 編碼
  archive/v1/                  歷史原始碼，不是執行依賴
```

`labtools` 不 import QICK 或 `QickworkspaceV2`。SDK 呼叫通用工具，但量測時序、
trace 選擇、實驗分析、final plot 和校正更新仍由 SDK 管理。安裝 workspace 會
同時安裝兩個套件。已移除舊工具路徑，沒有 V1 adapter 或轉接 import 模組。

## Notebook 與參數

日常使用 `NotebookLab`：Notebook 只 import、修改參數、呼叫 SDK。
`Measurement` 提供直接 Program 執行；`Session` 用於 schema 驗證與自動化。

```python
from QickworkspaceV2 import NotebookLab, QickSweep1D
from QickworkspaceV2.experiments.t1_ge import T1GEProgram

lab = NotebookLab("lab/project.yaml", qubit="Q1", calibration_enabled=False)
lab.connect()
qb = lab.config["Q1"]
run_cfg = qb.for_run(steps=81, wait_us=QickSweep1D("waitloop", 0, 100))
result = lab.run(T1GEProgram, run_cfg, py_avg=5)
lab.save_labber(result, directory="labber_data")
```

執行前核對 profile、gain 與儀器接線。`qb` 是工作設定的 live view，
`qb.for_run(...)` 複製目前工作值，再覆蓋本次參數，並保留原生 QICK sweep。
它不重新讀 YAML，也不量測。`qb.update(...)` 修改工作值；持久保存需要明確
save 或通過檢查的 `lab.apply_fit(result)`。頻率 MHz、時間 us、gain normalized。

各 GE／EF 實驗獨立維護。Program 明確呼叫 `setup_qubit_gen` 定義 qubit channel，
`setup_readout` 組合 resonator 與 readout；不再使用全包的 `setup_device`。
Punch-out 由 FPGA 的 gainloop／freqloop 掃描兩軸。寬頻使用獨立的
`broadband_resonator_spectrum`，不在窄頻 resonator 實驗中加入模式切換。

## LivePlot、分析與中斷

`NotebookLab.run()` 管理 live 圖、進度、每輪 partial checkpoint、分析及最後圖。
結束、中斷或失敗後清除自己的 live 顯示，保留 final plot。`LivePlot` 預設啟用
`tqdm`；圖像更新節流與進度計數分開。尚無資料的中斷繼續拋出；已有資料則保存
partial，再分析與顯示。fit 不通過不能丟棄成功取得的原始 IQ／shots。

公式與各模型起始值、bounds、衍生參數及品質判定在
`labtools/fitting/functions.py`；`solver.py` 不包含 model dispatch。
單次設定使用 `run_cfg["fit"]`，共用設定使用 `FIT_OPTIONS`。
`lab.analyze(result)` 重分析已保存資料，不重新量測；各實驗的 plot 與 updates
留在自己的 `analysis.py`。正式校正與工作設定套用均需明確呼叫且通過 fit 檢查。

## HDF5、Labber 與 SQL

原始 IQ、shots、coords、dims、units、metadata 保存在 HDF5。SDK 的
`runs.sqlite3` 是量測執行紀錄；分析另存版本，不覆寫不可變的 acquisition。

Labber 匯出可指定獨立 `LABBER_PATH`。目錄模式使用
`YYYY/MM/Data_MMDD/<experiment>_<target>_<kind>_<run_id>.hdf5`，並自動登錄
該根目錄的 `catalog.sqlite`。完整 native record 存在 `/metagroup`，final PNG
在 `/metagroup/plots/analysis.png`。Labber Completed 表示檔案匯出完成，
量測是否中斷仍以 acquisition status 為準。

```python
from labtools.catalog import find_experiments, rebuild_catalog
from labtools.hdf5 import read_hdf5

rows = find_experiments(data_root="labber_data", qubit="Q1", experiment="t1_ge")
report = rebuild_catalog("labber_data")
# 直接 native 匯出也可選擇登錄到相同類型的檔案索引：
result.save("exports/t1.h5", catalog_root="exports")
record, traces = read_hdf5("exports/t1.h5")
```

`files` 表以相對檔案路徑為主鍵，記錄 run ID、實驗、UTC 時間、targets、tags、
acquisition／analysis status、quality、格式、data kind 與註解。
不同 target、IQ、shots 匯出可以共存。索引不存量測陣列，可由檔案重建。
索引失敗會警告並保留 HDF5；重建會列出讀不到或不支援的檔案，不刪除它們。
這是 workspace 的檔案索引，不是 Labber GUI 自己的資料庫。

## Worker 與 Agent

UI、CLI、Agent 使用 worker 的 live `GET /catalog`；實驗清單及 schema 以
實際 registry 為準，不維護另一份 production catalog、wrapper 或 AST discovery。
`qick_agent/client.py` 是單一 HTTP client。worker 擁有硬體；companion 不 import QICK。

結果契約分開 acquisition status、analysis status 和 quality，避免壞 fit 被解讀為
需要重新量測。request journal、request ID 去重與唯讀 lookup 用於回應遺失時恢復；
UI 已移除獨立的手動 Recover job 面板。完整資料透過 run artifact 下載，compact
回應省略陣列不代表原始資料被刪除。詳見 [worker 文件](WORKER.md)。

## 驗證狀態

檢查指令見 [Verification](VERIFICATION.md)。最新工具拆分結果與尚未完成事項見
[labtools 驗證紀錄](../../docs/verification/2026-09-11-labtools.md)。較早日期的實機
紀錄只代表當時版本，不代表本次拆分或尚未完成搬遷的版本已通過實機驗證。
