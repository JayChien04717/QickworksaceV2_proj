# 驗證範圍

2026-09-10，檔案清理與 Notebook 英文化後重新驗證：**90 項測試通過，Ruff 通過**。主 Notebook 的 **17 組 program/config** 和根目錄 `measure.py` 已通過真正 QICK 0.2.422 compiler；兩本 notebook 通過英文內容、格式與程式語法檢查。**未連線或量測實體板卡**。

測試涵蓋 bulk config 更新、live working view 與獨立 run 副本、固定 qubit ID 選取、原生 QickSweep1D、GE/EF、三 qubit、MUX 子集合、實際量化座標、可變 sigma 的 GE→EF 銜接、multiple readout events、shots 正規化／多輪平均、取消後 partial 保存、fit failure、校正 revision、儀器恢復與服務 request ID。

Notebook 驗證只執行標記為 imports/configuration/definition/prepare 的格子，對設定呼叫真正 compiler；connection、acquire、calibration update 都略過。進階 notebook 做 syntax validation，實驗 program 由單元測試另外編譯。Transport 單元測試以明確的固定 IQ arrays 取代 RPC，常數資料必須被 fitting 品質檢查拒絕；沒有 runtime 模擬 backend。

先前已使用 NVIDIA 官方 discovery、runner、storage、models 驗證 **28 個 wrappers** 的 AST discovery、script validation、JSON arrays、HDF5/SQLite 與 PNG roundtrip。本次將該驗證工具移到 `tests/`，沒有重新下載或執行官方 core。合約測試使用明確標記的數值資料 fixture，不代表 agent 或板卡量測驗收。

報告：[release.json](verification/release.json)、[notebooks.json](verification/notebooks.json)、[blueprint.json](verification/blueprint.json)。

```powershell
.venv/Scripts/python -m pytest -q
.venv/Scripts/python tests/verify_notebooks.py
.venv/Scripts/ruff check QickworkspaceV2 tests lab measure.py
.venv/Scripts/python tests/verify_blueprint.py PATH_TO_BLUEPRINT/core
```

兩個測試警告來自 FastAPI/Starlette 使用的相依套件棄用通知。實體 bitstream、外部 LO/接線、晶片 pulse calibration、readout fidelity 仍需依 [HARDWARE.md](HARDWARE.md) 上機驗收；不能把 compile 通過當成物理量測通過。
