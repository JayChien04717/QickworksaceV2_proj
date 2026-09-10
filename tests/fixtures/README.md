# QICK compiler fixture

`qick_testbench.json` was obtained from the official QICK repository on 2026-09-10:
https://github.com/openquantumhardware/qick/blob/main/firmware/testbench/qick_testbench/soccfg.json

It describes a ZCU216 testbench. Tests make explicit logical copies of its logical generator/readout channels to exercise multi-qubit compilation. These copies do not describe a verified physical board or bitstream. The upstream license is included as `LICENSE.qick`.
