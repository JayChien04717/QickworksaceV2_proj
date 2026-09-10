# QICK Workspace 2.0

以原生 QICK tProc v2 編寫、多 qubit 的量測專案。支援直接 pulse 操作、`qb.x()`／`qb.halfx()`、fitting、plot、資料保存與 NVIDIA Calibration Agent 橋接。

**日常入口：[QickworkspaceV2.ipynb](QickworkspaceV2.ipynb)。直接量測腳本：[measure.py](measure.py)。**

完整說明：[重構、架構與 API 詳解](docs/REFACTOR_GUIDE.md)，包含 `for_run()`、`build()`、資料保存與各目錄用途。兩本正式 Notebook 使用英文。

## 開始量測

選 `.venv/Scripts/python.exe` 或 **QICK Workspace** notebook kernel。新環境：

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[hardware,service,notebook]"
.venv/Scripts/python -m ipykernel install --prefix .venv --name qickworkspace --display-name "QICK Workspace"
```

先在 [lab/project.yaml](lab/project.yaml) 選接線 profile，再核對 `hardware.yaml` 的通道、mixer 與 `device.yaml` 的 qubit 初始值。日常參數直接在 Notebook 或 [lab/config.py](lab/config.py) 編輯。

```python
from qick.asm_v2 import QickSweep1D
from QickworkspaceV2 import Measurement
from QickworkspaceV2.experiments.resonator_spec import ResonatorSpecProgram
from lab.config import CONNECTION, DATA_PATH, make_config

config_all = make_config()
qb = config_all["Q1"]
qb.update(res_gain_ge=0.15, res_sigma=0.01)

run_cfg = qb.for_run(
    steps=101,
    res_freq_ge=QickSweep1D("freqloop", 6710, 6725),
    relax_delay=0,
)
lab = Measurement.from_pyro4(**CONNECTION, data_path=DATA_PATH)
result = lab.run(ResonatorSpecProgram, run_cfg, py_avg=5)
result.plot()
print(result.metrics, result.path)
```

`qb.update(...)` 改工作設定；`qb.for_run(...)` 建立本次量測的獨立副本。全部 qubit 的共同值用 `config_all.update_all(reps=100, relax_delay=200)`。以固定 qubit ID 選取設定；`run_cfg` 支援一般字典更新，欄位使用 `res_sigma`、`wait_us`、`detuning_mhz`。

多 qubit：`config_all.for_run("Q1", "Q2", steps=81, ...)`，每顆的參數在 `run_cfg["qubits"]["Q2"]`。修改掃描範圍、pulse 或自訂參數不需要改 schema。頻率 MHz，時間 µs，gain 為 QICK normalized gain。

`lab.run()` 會直接量測、保存 acquisition 與編譯參數，再執行分析。`lab.compile(Program, run_cfg)` 只編譯。專案不提供模擬量測模式。

## 目錄

```text
QickworkspaceV2.ipynb       日常調參與逐格量測
measure.py                 可直接執行的單一量測腳本
lab/config.py              Python 工作設定
lab/project.yaml           連線、profile、資料位置
lab/profiles/direct|mux/   接線與初始 device 值
lab/procedures/            分開維護的 DC flux / TWPA 儀器流程
lab/blueprint_scripts/     28 個 NVIDIA wrappers
notebooks/advanced_measurements.ipynb  MUX、pulse、gate、coupler 與自動化
QickworkspaceV2/            SDK：device / programs / experiments / data / analysis / ...
tests/                     單元測試、Notebook 編譯與 Blueprint 驗證工具
docs/                      設定、開發、上機與橋接文件
archive/v1/                重建前原始程式與 notebook
```

**28 個實驗各自一個檔案**。`t1_ge.py`、`t1_ef.py`、`ramsey_ge.py` 等各自有完整 program 和分析入口，GE/EF 可以獨立維護。內建包含 TOF、readout g/e/f spectroscopy、GE/EF spectroscopy/Rabi/T1/Ramsey/echo、single shot、DRAG、repeated Rabi、AllXY、RB、joint tomography、conditional Ramsey、coupler chevron、resonator flux 與 TWPA probe。

## 進階與驗證

NVIDIA 使用 `qickworkspace serve` 連線到 QICK；`qickworkspace export-blueprint lab/blueprint_scripts` 只產生 wrappers，不連線。自動化仍使用有 schema 的 `Session`，與直接 Notebook 共用 QICK backend、資料、fit 和 plot 工具。

- [重構、設定與實驗編寫詳解](docs/REFACTOR_GUIDE.md)
- [硬體與讀出條件](docs/HARDWARE.md)
- [NVIDIA Blueprint](docs/BLUEPRINT.md)
- [驗證範圍](docs/VERIFICATION.md)

主 Notebook 的 17 個量測設定和 Python 腳本已用真正 QICK compiler 驗證；未對實體板卡執行 acquisition。初始接線與 pulse 值仍需依現場設備核對。
