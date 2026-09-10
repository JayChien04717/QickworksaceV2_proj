"""QICK compile/acquire boundary. Mapping follows declared ADC order, never qubit indices."""

from __future__ import annotations
import inspect
from threading import RLock
import numpy as np

from QickworkspaceV2.data.models import TraceData


class AcquisitionCancelled(RuntimeError):
    pass


def extract_coords(program, plan, target):
    coords = {}
    for sweep in plan.sweeps:
        if sweep.pulse:
            raw = program.get_pulse_param(sweep.pulse.format(target=target), sweep.parameter, as_array=True)
        else:
            raw = program.get_time_param(sweep.tag.format(target=target), sweep.parameter, as_array=True)
        values = np.asarray(raw, float).reshape(-1)
        if len(values) != sweep.points:
            raise ValueError(
                f"{target}/{sweep.name}: compiler returned {len(values)} coordinates, expected {sweep.points}"
            )
        if not np.isfinite(values).all() or len(np.unique(values)) != len(values):
            raise ValueError(f"{target}/{sweep.name}: sweep collapses at hardware resolution")
        coords[sweep.name] = values
    coords["readout"] = np.asarray(plan.readout_events)
    return coords


def unpack_iq(program, plan, averaged, raw=None):
    """QICK: list[ADC][readout, *sweeps, I/Q] → per-target named traces."""
    channels = list(program.ro_chs)
    if len(averaged) != len(channels):
        raise ValueError("Acquisition ADC count differs from program declaration")
    shape = tuple(s.points for s in plan.sweeps)
    dims = tuple(s.name for s in plan.sweeps) + ("readout",)
    nreads = len(plan.readout_events)
    traces = {}
    for target in plan.cfg["targets"]:
        qc = plan.cfg["qubits"][target]
        channel = qc["ro_ch"]
        index = channels.index(channel)
        arr = np.asarray(averaged[index])
        expected = (nreads, *shape, 2)
        if arr.shape != expected:
            raise ValueError(
                f"ADC {channel}: IQ shape {arr.shape}, expected {expected}; no dimensions are silently squeezed"
            )
        iq = np.moveaxis(arr[..., 0] + 1j * arr[..., 1], 0, -1)
        shots, shot_dims = None, ()
        if raw is not None:
            buf = np.asarray(raw[index])
            expected_raw = (program.reps, *shape, nreads, 2)
            if buf.shape != expected_raw:
                raise ValueError(f"ADC {channel}: raw shape {buf.shape}, expected {expected_raw}")
            shots = np.moveaxis(buf[..., 0] + 1j * buf[..., 1], 0, -2) / program.ro_chs[channel]["length"]
            shot_dims = (*dims[:-1], "shot", "readout")
        traces[target] = TraceData(
            iq,
            dims,
            extract_coords(program, plan, target),
            {s.name: s.unit for s in plan.sweeps},
            shots=shots,
            shot_dims=shot_dims,
            metadata={
                "adc_channel": channel,
                "readout_group": qc["readout_group"],
                "rotation_deg": qc["ro_phase"],
                "iq_offset_removed": False,
            },
        )
    return traces


class QICKBackend:
    def __init__(self, soc, soccfg, *, resource_id="qick"):
        self.soc, self.soccfg, self.resource_id = soc, soccfg, resource_id
        self.lock = RLock()

    @classmethod
    def from_pyro4(cls, ns_host, ns_port=8888, proxy_name="myqick"):
        from qick.pyro import make_proxy

        soc, soccfg = make_proxy(ns_host, ns_port, proxy_name)
        return cls(soc, soccfg, resource_id=f"{ns_host}:{ns_port}/{proxy_name}")

    def capabilities(self):
        cfg = self.soccfg
        return {
            "backend": "qick",
            "resource_id": self.resource_id,
            "qick_version": cfg.get("sw_version") if isinstance(cfg, dict) else cfg["sw_version"],
            "generators": cfg["gens"],
            "readouts": cfg["readouts"],
            "tprocs": cfg["tprocs"],
        }

    def validate(self, device, plan):
        if len(self.soccfg["tprocs"]) != 1 or self.soccfg["tprocs"][0].get("type") != "qick_processor":
            raise ValueError(
                "This backend requires one QICK tProc v2; cross-board synchronization is not implemented"
            )
        for q in plan.cfg["targets"]:
            qc = plan.cfg["qubits"][q]
            if qc["ro_ch"] >= len(self.soccfg["readouts"]):
                raise ValueError(f"{q}: ADC channel is absent from connected firmware")
        for port in device.hardware.generators.values():
            if port.channel >= len(self.soccfg["gens"]):
                raise ValueError(f"Generator {port.channel} absent from connected firmware")
            actual = self.soccfg["gens"][port.channel]
            if actual.get("has_mixer") and port.mixer_mhz is None:
                raise ValueError(f"Generator {port.channel} needs mixer_mhz")
        for group in plan.cfg["readout_groups"].values():
            cap = self.soccfg["gens"][group["port"]["channel"]]
            if group["mode"] == "mux":
                if not cap.get("n_tones"):
                    raise ValueError("Readout group needs a mux generator in this firmware")
                if max(m["tone_slot"] for m in group["members"].values()) >= cap["n_tones"]:
                    raise ValueError("tone_slot exceeds firmware mux capacity")
            elif cap.get("n_tones"):
                raise ValueError("Direct readout cannot use a mux-only generator")

    def compile(self, plan):
        program = plan.program(
            self.soccfg, reps=plan.cfg["reps"], final_delay=plan.cfg["relax_delay"], cfg=plan.cfg
        )
        for channel, envelopes in enumerate(program.envelopes):
            limit = self.soccfg["gens"][channel].get("maxlen")
            used = max(
                (envelope["addr"] + len(envelope["data"]) for envelope in envelopes["envs"].values()),
                default=0,
            )
            if limit is not None and used > limit:
                raise ValueError(
                    f"Generator {channel} needs {used} envelope samples; firmware supports {limit}; "
                    "shorten or reuse waveforms"
                )
        for memory in ("pmem", "wmem"):
            data = program.binprog.get(memory)
            limit = self.soccfg["tprocs"][0].get(memory + "_size")
            if data is not None and limit is not None and len(data) > limit:
                raise ValueError(
                    f"Program needs {len(data)} {memory} words; firmware supports {limit}; split the experiment"
                )
        return program

    def acquire(self, program, plan, *, soft_avgs, on_progress=None, cancel=None):
        if plan.metadata.get("acquisition_mode") == "decimated":
            return self._acquire_decimated(program, plan, soft_avgs, on_progress, cancel)
        total, all_shots = None, {}
        for index in range(soft_avgs):
            if cancel and cancel.is_set():
                raise AcquisitionCancelled(
                    "Cancelled between acquisition rounds; hardware acquisition has returned"
                )
            signature = inspect.signature(program.acquire).parameters
            kwargs = {"progress": False, "remove_offset": False}
            kwargs["rounds" if "rounds" in signature else "soft_avgs"] = 1
            averaged = program.acquire(self.soc, **kwargs)
            current = unpack_iq(program, plan, averaged, program.get_raw() if plan.capture_shots else None)
            if total is None:
                total = current
            else:
                for q in total:
                    total[q].iq += (current[q].iq - total[q].iq) / (index + 1)
            if plan.capture_shots:
                for q in total:
                    all_shots.setdefault(q, []).append(current[q].shots.copy())
                    total[q].shots = np.concatenate(all_shots[q], axis=-2)
            if on_progress:
                on_progress(index + 1, soft_avgs, total)
            if cancel and cancel.is_set():
                raise AcquisitionCancelled(
                    "Cancelled after acquisition round; available data saved as partial"
                )
        if plan.capture_shots:
            for q in total:
                total[q].shots = np.concatenate(all_shots[q], axis=-2)
        return total

    def _acquire_decimated(self, program, plan, soft_avgs, on_progress, cancel):
        if plan.sweeps or program.reps != 1 or len(plan.readout_events) != 1:
            raise ValueError("Decimated acquisition contract requires one rep/readout and no hardware sweeps")
        traces = None
        for average in range(soft_avgs):
            if cancel and cancel.is_set():
                raise AcquisitionCancelled("Decimated acquisition cancelled between rounds")
            arrays = program.acquire_decimated(self.soc, rounds=1, progress=False, remove_offset=False)
            if traces is None:
                traces = {}
            for q in plan.cfg["targets"]:
                qc = plan.cfg["qubits"][q]
                index = list(program.ro_chs).index(qc["ro_ch"])
                arr = np.asarray(arrays[index])
                if arr.ndim != 2 or arr.shape[-1] != 2:
                    raise ValueError(f"Unexpected decimated shape {arr.shape}")
                iq = (arr[:, 0] + 1j * arr[:, 1])[:, None]
                if average == 0:
                    traces[q] = TraceData(
                        iq,
                        ("time", "readout"),
                        {"time": program.get_time_axis(index), "readout": np.asarray(plan.readout_events)},
                        {"time": "us"},
                        metadata={
                            "adc_channel": qc["ro_ch"],
                            "rotation_deg": qc["ro_phase"],
                            "iq_offset_removed": False,
                        },
                    )
                else:
                    traces[q].iq += (iq - traces[q].iq) / (average + 1)
            if on_progress:
                on_progress(average + 1, soft_avgs, traces)
        return traces
