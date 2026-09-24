"""Hardware-independent node discovery and contract regression tests."""

import importlib
import inspect
import json
import unittest
from unittest.mock import patch

from QickworkspaceV2_agent.nodes.catalog import (
    experiment_schema, get_spec, list_experiments, resolve_experiment_class,
)


class AgentCatalogTests(unittest.TestCase):
    def test_catalog_preserves_original_ids_and_covers_new_families(self):
        original_ids = {
            "tof", "res_spec_ge", "res_punchout", "res_spec_flux", "qubit_spec_ge",
            "qubit_flux_spec_ge", "time_rabi_ge", "power_rabi_ge",
            "power_rabi_active_reset_ge", "drag", "aae", "ramsey_ge",
            "spin_echo_ge", "t1_ge", "res_spec_ef", "qubit_spec_ef", "power_rabi_ef",
            "ramsey_ef", "t1_ef", "allxy", "qubit_temp", "ac_stark", "rb",
            "rb_asm", "tomography",
        }
        specs = list_experiments()
        ids = {spec.id for spec in specs}
        self.assertEqual(len(ids), len(specs))
        self.assertTrue(original_ids <= ids)
        self.assertTrue({"single_shot", "single_shot_opt", "auto_rb", "mux_auto_rb",
                         "cryoscope_predistorted", "twpa_gain_power", "auxiliary_ef_spec"} <= ids)
        json.dumps(experiment_schema(), allow_nan=False)

    def test_resolvers_cover_public_runnable_exports_without_instantiation(self):
        from qick.asm_v2 import AveragerProgramV2
        from QickworkspaceV2_agent.core.base_analysis import BaseAnalysis

        classes = {resolve_experiment_class(spec.id) for spec in list_experiments()}
        for module_path in (
            "experiments", "experiments.setup", "experiments.resonator",
            "experiments.qubit_ge", "experiments.qubit_ef", "experiments.coherence",
            "experiments.characterization", "experiments.cryoscope",
            "experiments.cooling", "experiments.twpa", "experiments_mux",
        ):
            module = importlib.import_module(f"QickworkspaceV2_agent.{module_path}")
            for name in module.__all__:
                value = getattr(module, name)
                if (inspect.isclass(value) and callable(getattr(value, "run", None))
                        and not issubclass(value, (AveragerProgramV2, BaseAnalysis))):
                    with self.subTest(module=module_path, name=name):
                        self.assertIn(value, classes)
        for cls in classes:
            self.assertTrue(cls.__module__.startswith("QickworkspaceV2_agent."))

    def test_run_schemas_match_required_parameters_and_reject_unknown_options(self):
        for spec in list_experiments():
            with self.subTest(node=spec.id):
                cls = resolve_experiment_class(spec.id)
                method = cls.liveplot if spec.adapter == "mux_tof" else cls.run
                parameters = inspect.signature(method).parameters
                required = {
                    name for name, parameter in parameters.items()
                    if name != "self" and parameter.default is inspect.Parameter.empty
                    and parameter.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
                }
                self.assertTrue(required <= set(spec.run_schema["required"]))
                if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                    self.assertTrue(set(spec.run_schema["properties"]) <= set(parameters))
                self.assertFalse(spec.run_schema["additionalProperties"])

    def test_constructor_schema_matches_external_dependencies(self):
        for spec in list_experiments():
            if spec.adapter != "external_instrument":
                continue
            with self.subTest(node=spec.id):
                parameters = inspect.signature(resolve_experiment_class(spec.id)).parameters
                required = {name for name, parameter in parameters.items()
                            if name != "run_cfg" and parameter.default is inspect.Parameter.empty}
                self.assertEqual(required, set(spec.init_schema["required"]))
                self.assertEqual(spec.resource_kind, "external")
                self.assertEqual(spec.init_schema["properties"]["pump_source"]["x-resource"], "pump_source")

    def test_transition_updates_cannot_overwrite_ge_from_ef_results(self):
        expected = {
            "qubit_spec_ef": ("f0_MHz", "qb_freq_ef"),
            "power_rabi_ef": ("pi_gain", "pi_gain_ef"),
            "ramsey_ef": ("corrected_freq_MHz", "qb_freq_ef"),
            "t1_ef": ("T1_us", "T1_ef"),
        }
        for node, (result, target) in expected.items():
            with self.subTest(node=node):
                mapping = get_spec(node).update_map
                self.assertEqual(mapping[result], target)
                self.assertFalse(any(key.endswith("_ge") for key in mapping.values()))
        # The historical name describes an e-prepared resonator, not an EF drive.
        self.assertNotIn("f_res[MHz]", get_spec("res_spec_ef").update_map)
        # Multiplexed keys encode qubit names and cannot use a scalar update map.
        self.assertTrue(all(not spec.update_map for spec in list_experiments() if spec.category == "mux"))

    def test_discovery_does_not_import_and_mutation_does_not_change_catalog(self):
        with patch("QickworkspaceV2_agent.nodes.catalog.import_module") as importer:
            first = get_spec("t1_ge")
            experiment_schema()
            list_experiments()
            importer.assert_not_called()
            with self.assertRaises(ValueError):
                resolve_experiment_class("os.system")
            importer.assert_not_called()
        first.config_schema["properties"].clear()
        first.update_map["T1_us"] = "wrong"
        self.assertTrue(get_spec("t1_ge").config_schema["properties"])
        self.assertEqual(get_spec("t1_ge").update_map["T1_us"], "T1_ge")

    def test_active_reset_declares_threshold_dependency(self):
        for node in ("power_rabi_active_reset_ge", "ramsey_active_reset_ge"):
            self.assertIn("threshold", get_spec(node).requires)

    def test_portable_sweep_contract_passes_schema_and_gain_safety(self):
        from QickworkspaceV2_agent.nodes.contracts import Sweep
        from QickworkspaceV2_agent.nodes.validation import SafetyPolicy, validate_schema, validate_sweeps

        schema = get_spec("power_rabi_ge").config_schema
        for sweep in (Sweep("gainloop", 0.001, 0.08),
                      Sweep("gainloop", 0.001, 0.08, points=51, count_key="steps")):
            config = {"qb_gain_ge": sweep.to_dict()}
            validate_schema(config, schema)
            validate_sweeps(config)
            SafetyPolicy().validate(config)
        unsafe = {"qb_gain_ge": Sweep("gainloop", 0.001, 0.1, points=51).to_dict()}
        validate_schema(unsafe, schema)
        with self.assertRaises(ValueError):
            SafetyPolicy().validate(unsafe)


if __name__ == "__main__":
    unittest.main()
