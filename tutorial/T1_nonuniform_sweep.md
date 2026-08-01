# QICK tProc v2：用 DMEM 實作非均勻 T1 時間掃描

本筆記以 QICK 0.2.406、`AveragerProgramV2` 與本專案的
`BaseProgram` / `BaseExperiment` 為基礎，說明如何讓 T1 的等待時間呈現
「前面點密、後面點疏」的非均勻分布。

目標不是只把圖的 X 軸改成 log scale，而是讓硬體實際量測的時間點為
non-uniform spacing。

## 結論

原生 `QickSweep1D` 只支援固定增量：

```text
t[n] = start + n * step
```

若要使用平方分布、log spacing 或任意自訂時間點，可將每個等待時間轉成
tProc clock cycles，放進 DMEM，再由硬體迴圈逐點讀取：

```text
wait_times_us
    -> us2cycles()
    -> DMEM
    -> read_dmem()
    -> wait_cycles register
    -> TIME inc_ref
    -> readout
```

T1 掃描修改的是兩個操作之間的 reference time，不是 pulse waveform 的
length，因此這個功能不應透過 `w_length`、`read_wmem()` 或
`write_wmem()` 實作。

## 1. Macro 是什麼

QICK 的 `Macro` 是 Python 層的高階指令物件。它在建立 program 時被加入
`macro_list`，在 compile 階段再展開成一條或多條 tProc assembly
instructions。

```text
Python program definition
        -> Macro objects
        -> tProc assembly
        -> binary program
        -> FPGA execution
```

例如：

```python
self.delay(1.0)
```

不代表 Python 在此等待 1 us。它會先建立 `Delay(t=1.0)` macro，之後在
compile 階段轉成類似：

```text
TIME inc_ref, #<cycles>
```

### `preprocess()` 與 `expand()`

較完整的 Macro 可以有兩個階段：

- `preprocess(prog)`：配置 register、進行單位轉換、處理 timeline 或 sweep。
- `expand(prog)`：輸出實際的低階 `AsmInst`。

本筆記的等待時間已經在 `compile_datamem()` 轉成 clock cycles，而且使用的
register 也已經事先宣告，所以 custom Macro 只需要實作 `expand()`。

## 2. PMEM、DMEM、WMEM 與 register

| 名稱 | 用途 |
| --- | --- |
| PMEM | 儲存 tProc program instructions |
| DMEM | 儲存 32-bit 資料；本例用來放等待時間表 |
| 一般 data register `rN` | 暫存 DMEM address、等待時間或運算結果 |
| WMEM | 儲存 pulse waveform descriptor，例如 frequency、gain 和 length |
| waveform registers `w0...w5` | 修改一筆 waveform descriptor 時的工作 register |
| special registers `sN` | tProc 的時間、控制、port 等特殊功能 |

本例只需要：

```text
DMEM + 一般 data registers + TIME instruction
```

不需要修改 WMEM。

### 兩個一般 register

```python
self.add_reg("table_index")
self.add_reg("wait_cycles")
```

QICK compiler 會替名稱配置實際的 `rN`。使用者不需要假設
`table_index` 一定是 `r0`。

```python
self.read_dmem(
    dst="wait_cycles",
    addr="table_index",
)
```

概念上等於：

```python
wait_cycles = DMEM[table_index]
```

`read_dmem()` 不會自動增加 address，因此每個量測點結束後要執行：

```python
self.inc_reg("table_index", 1)
```

## 3. 如何選擇非均勻時間點

### 3.1 平方分布：一般 T1 的建議起點

```python
u = np.linspace(0.0, 1.0, steps)
wait_times_us = max_wait_us * u**2
```

如果 `max_wait_us=100`、`steps=11`，會得到：

```text
0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 us
```

一般形式：

```python
wait_times_us = max_wait_us * np.linspace(0, 1, steps)**power
```

| `power` | 分布 |
| --- | --- |
| 1.0 | 線性 |
| 1.5 | 前段稍密 |
| 2.0 | 前段明顯較密，推薦起點 |
| 3.0 | 大量點集中於前段 |

### 3.2 Log spacing

`geomspace()` 不能從 0 開始，所以零點需要另外加入：

```python
wait_times_us = np.concatenate([
    [0.0],
    np.geomspace(0.05, 100.0, steps - 1),
])
```

Log spacing 比較適合時間範圍跨越多個數量級的情況。若時間範圍不大，平方
分布通常更容易控制。

### 3.3 任意自訂點

```python
wait_times_us = np.array([
    0.0,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
])
```

只要時間非負、DMEM 放得下，table-driven 方法不要求任何特定數學形式。

## 4. Custom time Macro

QICK 0.2.406 的 `delay()` / `delay_auto()` 接受 float 或 `QickParam`，沒有
提供直接傳入使用者命名 register 的公開介面。因此需要一個很小的 custom
Macro：

```python
from qick.asm_v2 import AsmInst, Macro


class DelayFromRegister(Macro):
    """以指定 register 的 clock cycles 增加 tProc reference time。"""

    def expand(self, prog):
        return [
            AsmInst(
                inst={
                    "CMD": "TIME",
                    "C_OP": "inc_ref",
                    "R1": prog._get_reg(self.reg),
                },
                addr_inc=1,
            )
        ]
```

建立 Macro：

```python
DelayFromRegister(reg="wait_cycles")
```

compile 時，`prog._get_reg("wait_cycles")` 會把名稱解析成實際的 `rN`，產生
概念上如下的 assembly：

```text
TIME inc_ref, rN
```

也就是：

```text
reference_time += wait_cycles
```

注意：`_get_reg()` 是 QICK 內部 API；若日後升級 QICK，應重新確認名稱與
行為。

## 5. 完整 T1 non-uniform implementation

建議新增檔案：

```text
QickworkspaceV2/experiments/coherence/t1_nonuniform.py
```

以下是可整合進目前框架的完整範例：

```python
"""T1 relaxation measurement with a DMEM-backed non-uniform time axis."""

from __future__ import annotations

import numpy as np

from qick.asm_v2 import AsmInst, AsmV2, Macro

from ...analysis.qubit import T1Analysis
from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram


class DelayFromRegister(Macro):
    """Increment tProc reference time by the cycles stored in a register."""

    def expand(self, prog):
        return [
            AsmInst(
                inst={
                    "CMD": "TIME",
                    "C_OP": "inc_ref",
                    "R1": prog._get_reg(self.reg),
                },
                addr_inc=1,
            )
        ]


class T1NonuniformProgram(BaseProgram):
    """T1 program using a DMEM lookup table for the wait time."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")

        # General-purpose tProc registers.
        self.add_reg("table_index")
        self.add_reg("wait_cycles")

        # This loop defines acquisition shape only. It is not tied to a
        # QickParam or QickSweep1D.
        self.add_loop("waitloop", len(cfg["wait_times_us"]))

        self.setup_qb_pulse(
            cfg,
            prefix="ge",
            name="qb_pulse",
            gain_key="pi_gain_ge",
        )

    def compile_datamem(self):
        """Compile wait_times_us into 32-bit tProc core-clock cycles."""
        requested_times = np.asarray(
            self.cfg["wait_times_us"],
            dtype=float,
        )

        if requested_times.ndim != 1:
            raise ValueError("wait_times_us must be one-dimensional")
        if requested_times.size == 0:
            raise ValueError("wait_times_us cannot be empty")
        if not np.all(np.isfinite(requested_times)):
            raise ValueError("wait_times_us must contain only finite values")
        if np.any(requested_times < 0):
            raise ValueError("T1 wait times cannot be negative")
        if np.any(np.diff(requested_times) < 0):
            raise ValueError("wait_times_us must be monotonically increasing")

        # No gen_ch/ro_ch: delay uses the tProc core clock.
        wait_cycles = np.asarray(
            [
                self.us2cycles(us=float(wait_us))
                for wait_us in requested_times
            ],
            dtype=np.int64,
        )

        if np.any(wait_cycles > np.iinfo(np.int32).max):
            raise ValueError("wait time exceeds the 32-bit tProc time range")

        # Store the hardware-rounded values for plotting, fitting and saving.
        self.wait_times_actual_us = np.asarray(
            [
                self.cycles2us(cycles=int(cycles))
                for cycles in wait_cycles
            ],
            dtype=float,
        )

        return wait_cycles.astype(np.int32)

    def _body(self, cfg):
        # wait_cycles = DMEM[table_index]
        self.read_dmem(
            dst="wait_cycles",
            addr="table_index",
        )

        self.send_readoutconfig(
            ch=cfg["ro_ch"],
            name="myro",
            t=0,
        )

        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        # Prepare |e>.
        self.pulse(
            ch=cfg["qb_ch"],
            name="qb_pulse",
            t=0,
        )

        # First align the reference time to the end of the qubit pulse and
        # preserve the original fixed 0.05 us guard interval.
        self.delay_auto(
            t=0.05,
            tag="guard",
        )

        # Add this sweep point's arbitrary wait time from DMEM.
        self.append_macro(
            DelayFromRegister(reg="wait_cycles")
        )

        self.measure(cfg)

        # Advance to the next DMEM word.
        self.inc_reg(
            dst="table_index",
            src=1,
        )


class T1Nonuniform(BaseExperiment):
    """T1 relaxation measurement with arbitrary non-uniform wait times."""

    EXPT_NAME = "s008b_T1_nonuniform_ge"
    TAG = "T1"
    X_LABEL = "Delay time (us)"
    TITLE_PREFIX = "Qubit T1 ge - nonuniform"
    SWEEP_KEYS_TO_REMOVE = ["wait_times_us"]

    X_SAVE_NAME = "Delay time"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    Analysis = T1Analysis

    def _create_program(self):
        # AveragerProgramV2 normally runs:
        # for rep in reps:
        #     for wait point in waitloop:
        #         body()
        # Reset the DMEM index at the start of every hardware rep.
        reset_table = AsmV2()
        reset_table.write_reg(
            dst="table_index",
            src=0,
        )

        return T1NonuniformProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            before_reps=reset_table,
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        # The delay is no longer a QickParam, so get_time_param() is not the
        # correct source. Use values reconstructed from integer clock cycles.
        return prog.wait_times_actual_us

    def _save_comment(self, dict_val):
        if self.result is not None:
            t1_us = self.result.fit_result.get("T1_us", (None,))[0]
            if t1_us is not None:
                return f"T1 = {t1_us:.2f} us\n{dict_val}"
        return str(dict_val)
```

若要讓它成為 package API，還需要在下列檔案 export：

```text
QickworkspaceV2/experiments/coherence/__init__.py
QickworkspaceV2/experiments/__init__.py
QickworkspaceV2/__init__.py
```

實際需要 export 到哪一層，依預期的 import path 決定。

## 6. Notebook 使用方式

### 平方分布

```python
import numpy as np

from QickworkspaceV2 import ExperimentConfig
from QickworkspaceV2.config.system_cfg import config_list
from QickworkspaceV2.experiments.coherence.t1_nonuniform import T1Nonuniform


cfg = ExperimentConfig(config_list).get_qubit("Q1")

steps = 51
max_wait_us = 100.0
power = 2.0

cfg["wait_times_us"] = (
    max_wait_us * np.linspace(0.0, 1.0, steps) ** power
)

result = T1Nonuniform(cfg).run(py_avg=5)
```

### Log spacing

```python
steps = 51

cfg["wait_times_us"] = np.concatenate([
    [0.0],
    np.geomspace(0.05, 100.0, steps - 1),
])

result = T1Nonuniform(cfg).run(py_avg=5)
```

這個 implementation 使用：

```python
len(cfg["wait_times_us"])
```

作為實際 loop count，因此不依賴 `cfg["steps"]`；可避免 `steps` 與 table
長度不同步。

## 7. 為什麼每個 rep 都要 reset `table_index`

預設 loop 結構大致如下：

```python
for rep in range(reps):
    for point in range(len(wait_times_us)):
        body()
```

第一個 rep 跑完後，`table_index == len(wait_times_us)`。若第二個 rep 沒有重設，
就會讀到 DMEM table 以外的 address。

因此使用：

```python
reset_table = AsmV2()
reset_table.write_reg("table_index", 0)
```

並透過：

```python
before_reps=reset_table
```

讓 index 在每個 hardware rep 開始時回到 0。

## 8. `delay_auto(0.05)` 與 table delay 的分工

兩個 delay 有不同用途：

```python
self.delay_auto(0.05)
```

負責把 reference time 移到 qubit pulse 結束後，再增加固定 0.05 us guard。

```python
self.append_macro(DelayFromRegister(reg="wait_cycles"))
```

負責增加這個 sweep point 從 DMEM 讀出的任意等待時間。

實際序列為：

```text
qubit pi pulse
    -> pulse end
    -> fixed 0.05 us guard
    -> wait_times_us[i]
    -> readout
```

因此 table 中的 0 us 仍然保留 0.05 us guard。若將 guard 算進物理上的總自由
演化時間，可把 X 軸改為：

```python
return prog.wait_times_actual_us + 0.05
```

但在所有點都有相同固定 offset 的情況下，指數模型的 T1 decay constant 通常
不會改變；需要在整個專案內統一 X 軸定義即可。

## 9. 為什麼使用硬體 rounding 後的 X 軸

tProc time 只能是整數 clock cycles。例如要求：

```text
0.012345 us
```

硬體實際執行值可能是：

```text
0.012352 us
```

因此 fit、plot 和 save 應使用：

```python
prog.wait_times_actual_us
```

而不是直接使用原始的：

```python
cfg["wait_times_us"]
```

轉換流程為：

```text
requested microseconds
    -> us2cycles()
    -> integer tProc cycles
    -> cycles2us()
    -> actual microseconds
```

現有 `T1Analysis` 直接使用 `ExperimentData.x_axis` 做 exponential fit，因此只要
`_extract_sweep_axis()` 回傳正確的 non-uniform actual times，fit 本身不需要
假設等間距，也不需要修改。

## 10. DMEM 容量與限制

每個等待時間使用一個 32-bit DMEM word。可由以下設定檢查容量：

```python
dmem_size = soccfg["tprocs"][0]["dmem_size"]
```

必須滿足：

```python
len(wait_times_us) <= dmem_size
```

如果日後還有其他功能使用 DMEM，需替不同 table 分配 address offset，不能都從
address 0 開始。

## 11. 建議驗證順序

1. 使用 5 至 10 個容易辨識的時間點，例如 `[0, 1, 2, 5, 10]` us。
2. 建立 experiment 後先呼叫 `prog_asm()`，確認 assembly 中包含 DMEM read、
   `TIME inc_ref` 與 index increment。
3. 確認 `prog.wait_times_actual_us` 單調遞增且長度正確。
4. 先使用低 `reps` / `py_avg` 做硬體 smoke test。
5. 檢查每個 rep 都從 DMEM address 0 開始。
6. 和原本線性 T1 在相同時間範圍比較，確認 decay 與 fit 的 T1 相容。
7. 確認 live plot、儲存的 X 軸與 raw IQ point count 完全一致。

範例：

```python
cfg["wait_times_us"] = np.array([0, 1, 2, 5, 10], dtype=float)

expt = T1Nonuniform(cfg)
prog = expt.prog_asm()

print(prog.wait_times_actual_us)
```

## 12. 常見錯誤

### 把等待時間寫入 `w_length`

`w_length` 是 waveform/pulse segment 的長度，不是 pulse 之後的 T1 free-evolution
delay。修改它會改變 pulse 本身。

### 把微秒直接寫進 DMEM

DMEM 儲存的是 integer word：

```python
# 錯誤
return np.asarray(wait_times_us)

# 正確
return np.asarray([
    self.us2cycles(us=t)
    for t in wait_times_us
], dtype=np.int32)
```

### 忘記 reset table index

第一個 rep 正常、第二個 rep 開始資料異常，通常就是 index 沒有在
`before_reps` 歸零。

### 使用 `get_time_param("wait", ...)`

table delay 不是 `QickParam`，所以不能再由 `get_time_param()` 推導 sweep axis。
應直接保存 cycles 換算後的實際時間。

### `steps` 與 table 長度不同

loop count 應直接使用：

```python
len(cfg["wait_times_us"])
```

避免同時維護兩個可能不一致的設定。

### 時間點過度集中在零附近

T1 fit 需要涵蓋明顯 decay 的區域。已知大約 T1 時，建議：

```python
max_wait_us = 4 * expected_t1_us
```

至：

```python
max_wait_us = 5 * expected_t1_us
```

可先從 `power=1.5` 或 `power=2.0` 開始，不宜一開始就使用過大的 power。

## 13. 實作狀態

本文件提供的是針對 QICK 0.2.406 與目前 QickworkspaceV2 架構設計的完整實作
筆記，尚未把 `T1Nonuniform` 實際加入 package source 或 export。正式整合後仍需
在實際 tProc v2 hardware 上完成 assembly inspection 與 hardware-in-the-loop
驗證。

QICK `asm_v2` API 可參考：

- <https://docs.qick.dev/latest/_autosummary/qick.asm_v2.html>
- <https://docs.qick.dev/latest/_modules/qick/asm_v2.html>
