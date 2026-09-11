# QICK Workspace 2.0

原生 QICK tProc v2 多 qubit 量測 SDK。晶片校正入口：[ChipCalibration.ipynb](ChipCalibration.ipynb)。Notebook 只需 SDK import、參數設定與呼叫；SDK 管理 acquisition、資料保存、中斷復原、分析和最後圖形。

See the [experiment file and parameter guide](QickworkspaceV2/experiments/README.md) to find where to edit parameters, pulse sequences, analysis, plots and calibration updates.

## 開始量測

使用 `qick_gui` 或安裝 SDK 的 Notebook kernel。新環境安裝 `pip install -e ".[hardware,service,notebook]"`。先在 [lab/project.yaml](lab/project.yaml) 選接線 profile，再核對 hardware/device 設定。

```python
from QickworkspaceV2 import NotebookLab, QickSweep1D
from QickworkspaceV2.notebook import enable_notebook
from QickworkspaceV2.experiments.t1_ge import T1GEProgram

enable_notebook()
lab = NotebookLab("lab/project.yaml", qubit="Q1", calibration_enabled=False)
lab.connect()
qb = lab.config["Q1"]
run_cfg = qb.for_run(
    steps=81, wait_us=QickSweep1D("waitloop", 0, 100),
    sigma_ge=0.05, relax_delay=200,
    fit={"p0": {"tau": 25}, "bounds": {"tau": (0.1, 200)}},
)
result = lab.run(T1GEProgram, run_cfg, py_avg=5)
```

`qb.update(...)` 更新工作設定；`qb.for_run(...)` 建立獨立的本次量測字典。`lab.config.update_all(...)` 明確更新所有 qubit，多 qubit 用 `lab.config.for_run("Q1", "Q2", ...)`。日常修改不需要改 node schema。頻率 MHz，時間 us，gain 為 QICK normalized gain。

所有實驗的 `lab.run()`／`lab.scan()` 預設顯示 LivePlot；完成、中斷或失敗後清除暫時的 live 圖，只保留最後分析圖與文字訊息。中斷後若已有資料，SDK 保存並分析目前 partial；尚無資料則中斷例外繼續傳出。`lab.analyze(result)` 重讀已存資料重新分析，不重新量測。`lab.scan()` 管理 host loop 與累積結果，readout frequency 自動讀取編譯後的 direct/MUX 頻率，不需填 coordinate callback。

每輪仍保存 `partial.h5`，中斷後可保留並分析已取得的資料。進階程式可直接使用 `Measurement.run()`／`Measurement.compile()`；沒有另一套量測 backend 或模擬模式。

## 目錄

```text
ChipCalibration.ipynb       日常晶片校正：只 import、改參數、呼叫 SDK
QickworkspaceV2.ipynb        原 Notebook 與已記錄輸出
measure.py                  直接量測腳本
lab/config.py               Python 工作設定
lab/project.yaml            連線與 profile
lab/profiles/               接線與初始 device 值
lab/procedures/             DC flux / TWPA 儀器流程
notebooks/advanced_measurements.ipynb   MUX、pulse、gate、coupler 與自動化
QickworkspaceV2/experiments/t1_ge/
    program.py              原生 Program 與 build(ctx, parameters)
    parameters.py           此實驗的 node schema、預設值與驗證
    analysis.py             analyze、plot 與 updates 校正規則
    __init__.py             公開 import 與 ExperimentSpec 註冊
QickworkspaceV2/analysis/fitting.py      公式／fit 函式配對與共用求解器
QickworkspaceV2/notebook.py              NotebookLab 執行與顯示
archive/v1/                 重建前原始程式與 Notebook
```

**35 個 catalog 實驗各有自己的資料夾**，GE／EF 獨立維護。Notebook 與 worker 使用相同 Program、分析和繪圖。一般一維圖使用共用 renderer；RB、tomography、readout、reset 等專屬圖放在實驗的 analysis.py。沒有舊檔案 wrapper 或相容層。

`broadband_resonator_spectrum` 是獨立的寬頻實驗，擁有自己的 `program.py`、`parameters.py`、`analysis.py` 與 catalog 項目；`resonator_spec` 維持單共振器窄頻分析。寬頻沿用原本 `analysis/fit_n_res.py` 的平滑、凹峰評分、phase reference 與三點二次插值，Notebook 使用 `BroadbandResonatorSpecProgram` 和通用 `lab.run()`，可編輯 `count`、`detection_options`、`y_mode`。

## 晶片校正與 fitting

Notebook 包含 TOF、寬頻/窄頻 resonator、punchout、GE/EF spectroscopy、time/power Rabi、single shot、readout gain/length 最佳化、dispersive、T1/Ramsey/echo、GE 頻率修正、g/e/f readout、temperature、repeated Rabi、DRAG、AllXY、RB/interleaved RB、tomography、feedback reset 與 CKP map。外部 flux 預設關閉。

各 tuning cell 可修改 sigma、pulse type、length、gain、threshold、rotation、reps 與掃描範圍。TOF 的 `check_e`／`check_f` 擇一開啟；`tof_threshold` 是波形包絡門檻，`ro_threshold` 是 single-shot 分類門檻。

公式集中在 [analysis/fitting.py](QickworkspaceV2/analysis/fitting.py)，每個公式都有自己的 fitting 函式，例如 `exponential()`／`fit_exponential()`。初始值、bounds、衍生參數與品質判定在對應 `fit_*()` 裡修改；`fit_curve()` 只接收公式 callable 做共用數值求解。`FIT_OPTIONS` 可覆寫各模型設定。單次量測用 `run_cfg["fit"]` 覆寫 model、具名 p0/bounds、maxfev、min_r_squared。修改共用公式／設定後，使用 `lab.analyze(result)` 重新分析已存資料。

校正寫回須先明確設定 `lab.calibration_enabled = True`，再執行 `lab.apply_fit(result)`。更新哪些參數由各實驗 `analysis.py` 的 `updates(result, target)` 宣告；Notebook 不再提供 fit 到設定的對應表。掃描圖、最佳 readout／DRAG 選點及 Ramsey 雙頻修正也由各實驗管理。未完成或未通過分析的結果不會更新校正；目前無晶片驗證維持停用。

## 自動化與驗證

UI／CLI／Agent 統一讀取 worker `GET /catalog`，不產生 wrappers。`Session` 使用 schema，Notebook 使用 native dict，兩者共用 Program、backend、資料與分析。改動已註冊實驗後重啟 worker 與 Agent backend。

- [Notebook API 與實驗結構](docs/NOTEBOOK_API.md)
- [進階設定與原生 QICK 開發](docs/REFACTOR_GUIDE.md)
- [硬體與讀出條件](docs/HARDWARE.md)
- [V2 worker 與 Agent](docs/WORKER.md)
- [驗證範圍](docs/VERIFICATION.md)

軟體檢查：`.venv/Scripts/python -m pytest -q`、`.venv/Scripts/python tests/verify_notebooks.py`、`.venv/Scripts/ruff check QickworkspaceV2 tests lab measure.py`。Notebook compiler 檢查不連線，實體驗證另用 qick_gui。日期紀錄放 repository 的 docs/ 與 data/verification/。
