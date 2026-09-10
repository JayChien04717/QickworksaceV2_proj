# Hardware contract 與上機驗收

## 已驗證範圍

Python 3.12、QICK 0.2.422。測試使用**真正的 QICK compiler**，config 來自 [官方 ZCU216 testbench](https://github.com/openquantumhardware/qick/blob/main/firmware/testbench/qick_testbench/soccfg.json)，再以測試程式擴充 logical channels。這是 compiler fixture，不代表實際 bitstream 或板卡已認證。

已檢查 direct、MUX、tProc/PYNQ readout、GE/EF preparation、原生一維／二維 sweeps、ADC declaration order、多次 readout。SDK 可讀 capabilities，但無法推斷外部線接到哪顆 qubit。本次沒有實體 acquisition。

## 單位

- frequency_mhz 使用 QICK `ABSOLUTE_FREQS` 語意。外接 LO 的 RF 頻率需作者明確換算，不能直接把它等同 laboratory RF frequency。
- mixer_mhz 是 generator 的 RFDC mixer；有 mixer 的 generator 必填。
- 時間統一 µs；gain 為 QICK v2 normalized DAC gain [-1,1]，同時受 port max_gain 限制，不是外部 DC voltage。
- Generator 與 digital readout channel 是不同 namespace。PFB endpoints 可共用實體 ADC，但同次 run 的 qubits 不能共用同一 digital endpoint。
- Direct readout helper 支援 const、arb、flat_top；後兩者由 res_sigma 設定 envelope。MUX 使用固定 tone list 和 constant mask。

## Bring-up

使用 [QickworkspaceV2.ipynb](../QickworkspaceV2.ipynb)。先核對 logical channels、mixer、trigger pins、readout timing；`lab.compile(Program, run_cfg)` 只編譯。再做低 gain TOF、readout resonance、spectroscopy、Rabi，之後校正 classifier。

樣本 profile 有示意 C12 port；沒有該 generator 時請移除 C12，別隨意映射到其他 drive。部分 GE/readout frequency seed 取自舊例子，其他數值為示意；全部都不是本次實測校正。

## Acquisition 與 timing

同時 gates 在同一 QICK program。一般 helper gate 後推進 timeline；parallel layer 結束後等待所有 pulses/readouts。

預設 reps loop 在 sweep loops 外。QICK averaged input 是每個 ADC 的 `[readout,*sweeps,IQ]`；raw 是 `[reps,*sweeps,readout,IQ]`。Shots 除以 integration length，offset removal 明確關閉，與平均 IQ 一致。

TOF 使用一個 hardware rep/readout，加 software averages；buffer 上限由 QICK 驗證。主 Notebook 的 direct/tProc readout spectroscopy 使用 QickSweep1D 硬體掃描。Session 的 readout spectroscopy 採 host sweep，支援 static MUX tone/PYNQ readout；兩者保存實際量化頻率。Echo 保存兩段 delay 的實際總時間，可能與要求終點略有量化差異。

## 限制

本版支援單一 tProc v2 owner。預檢 program/waveform memory、tone capacity；可操作 qubit 數仍取決於 firmware、頻寬、接線及 endpoints。未實作跨板 deterministic synchronization、FPGA feedback reset、surface-code cycles 或自動 CZ optimum 搜尋。

Worker 序列執行量測，OS lease 保護同一 resource_id。Notebook 和 agent 同時工作時共用 worker。Cancel 是 round/host-point 間的 cooperative cancellation，不是 FPGA emergency stop；RPC timeout、Ctrl-C、process crash 後不能宣稱 FPGA 已停止，需依現場復原流程處理。

Raw、partial data、failed run journal 均保存供診斷。完成實際板卡與晶片驗收後，才把 profile 視為正式量測設定。

Gate helper 與 GE→EF state preparation 使用 `wait_pulses()`，在 pulse/readout 結束後保留一個 tProc timing tick，避免不同時鐘的量化造成重疊。原生 `delay_auto()` 語意不變；精細時序仍由使用者的 program 控制。
