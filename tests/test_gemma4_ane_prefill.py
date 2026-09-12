# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 hybrid ANE prefill: the GeGLU seam and the Gemma-side gating.

The runtime itself is covered by ``test_qwen35_ane_prefill.py``; that file
passing unchanged is the regression gate on the shared ``geglu`` field. What
is tested here is what Gemma 4 adds: one activation functor reached by two
merge kernels, one defaulted parameter through the native entry points, and
a thin enable path that refuses MoE and CPU-shared combinations.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import omlx.patches.gemma4_ane_prefill as gemma4_patch
import omlx.patches.qwen35_ane_prefill as ane_patch

_CSRC = (
    Path(__file__).resolve().parents[1]
    / "omlx/custom_kernels/qwen35_prefill/csrc"
)
_SWIGLU_EXPR = "gate * up / (1.0f + exp(-gate))"


def _metal() -> str:
    return (_CSRC / "qwen35_ane.metal").read_text(encoding="utf-8")


# --- Native seam -----------------------------------------------------------


def test_each_activation_is_written_exactly_once():
    """No merge kernel keeps a forked copy of an activation expression."""
    metal = _metal()
    assert metal.count(_SWIGLU_EXPR) == 1
    assert metal.count("0.7978845608028654f") == 1
    # Every activation site goes through the functor template parameter.
    assert metal.count("Act()(gate, up)") == 6
    assert metal.count("struct SwiGlu {") == 1
    assert metal.count("struct GeGlu {") == 1


def test_split_merge_reads_gate_and_up_from_separate_buffers():
    """The split merge indexes two suffix buffers at one projection's stride."""
    metal = _metal()
    body = metal.split("void qwen35_ane_merge_glu_split_output(", 1)[1].split(
        "\n}", 1
    )[0]
    assert "gpu_gate[base + suffix]" in body
    assert "gpu_up[base + suffix]" in body
    # One projection per buffer, so the row stride is gpu_hidden, where the
    # packed merge strides by 2 * gpu_hidden over both halves.
    assert "m * static_cast<uint>(gpu_hidden)" in body
    packed = metal.split("void qwen35_ane_merge_glu_output(", 1)[1].split("\n}", 1)[0]
    assert "m * static_cast<uint>(2 * gpu_hidden)" in packed
    # The activation still goes through the shared functor.
    assert "Act()(gate, up)" in body


def test_geglu_functor_is_the_canonical_tanh_gelu():
    """gelu_approx(gate) * up, matching mlx_lm.models.gemma4_text.geglu."""
    metal = _metal()
    body = metal.split("struct GeGlu {", 1)[1].split("};", 1)[0]
    assert "0.7978845608028654f * (gate + 0.044715f * gate * gate * gate)" in body
    assert "up * 0.5f * gate * (1.0f + tanh(inner))" in body


def test_only_non_cpu_merges_are_instantiated_for_geglu():
    """CPU sharing and fused-down stay SwiGLU-only, with no dead kernels."""
    metal = _metal()
    for name in (
        "qwen35_ane_merge_geglu_output_",
        "qwen35_ane_merge_dual_geglu_output_",
        "qwen35_ane_merge_geglu_split_output_",
    ):
        assert f'instantiate_kernel("{name}" #type' in metal
    assert "GeGlu)" in metal
    assert metal.count("GeGlu)") == 3
    # The three out-of-scope kernels take the parameter but stay SwiGLU.
    for template in (
        "qwen35_ane_merge_dual_cpu_glu_output, type, SwiGlu)",
        "qwen35_ane_merge_cpu_glu_output, type, SwiGlu)",
        "qwen35_ane_glu_suffix, type, SwiGlu)",
    ):
        assert template in metal


def test_existing_host_kernel_names_are_unchanged():
    """Renaming the templates must not move the .mm lookup names."""
    metal = _metal()
    mm = (_CSRC / "qwen35_ane.mm").read_text(encoding="utf-8")
    for name in (
        "qwen35_ane_merge_swiglu_output_",
        "qwen35_ane_merge_dual_swiglu_output_",
        "qwen35_ane_merge_cpu_swiglu_output_",
        "qwen35_ane_merge_dual_cpu_swiglu_output_",
        "qwen35_ane_swiglu_suffix_",
    ):
        assert f'instantiate_kernel("{name}" #type' in metal
        assert f'"{name}"' in mm


def test_geglu_participates_in_primitive_identity():
    """A cached SwiGLU primitive must not be reused for a GeGLU call."""
    mm = (_CSRC / "qwen35_ane.mm").read_text(encoding="utf-8")
    for start, end, kernel in (
        (
            "class AneHybridQ4Primitive",
            "class DualAneHybridPrimitive",
            "qwen35_ane_merge_geglu_output_",
        ),
        (
            "class DualAneHybridPrimitive",
            "class AneHybridQ4SwiGLUDownPrimitive",
            "qwen35_ane_merge_dual_geglu_output_",
        ),
    ):
        block = mm.split(start, 1)[1].split(end, 1)[0]
        assert "bool geglu_;" in block
        assert "geglu_(geglu)" in block
        assert "geglu_ == rhs.geglu_" in block
        assert "fuse_swiglu_, geglu_," in block
        assert f'"{kernel}"' in block


def test_geglu_leaves_the_output_width_arithmetic_alone():
    """The flag selects a kernel name; fuse_swiglu_ still sizes the merge."""
    mm = (_CSRC / "qwen35_ane.mm").read_text(encoding="utf-8")
    for start, end in (
        ("class AneHybridQ4Primitive", "class DualAneHybridPrimitive"),
        ("class DualAneHybridPrimitive", "class AneHybridQ4SwiGLUDownPrimitive"),
    ):
        block = mm.split(start, 1)[1].split(end, 1)[0]
        widths = [
            line for line in block.splitlines() if "const int merge_" in line
        ]
        assert widths
        for line in widths:
            assert "geglu_" not in line
            assert "fuse_swiglu_ ?" in line


def test_fused_entry_points_gained_a_defaulted_parameter_not_new_symbols():
    """Source- and Python-compatible: existing callers stay untouched."""
    header = (_CSRC / "qwen35_ane.h").read_text(encoding="utf-8")
    bindings = (_CSRC / "bindings.cpp").read_text(encoding="utf-8")
    for name in (
        "qwen35_ane_q4_swiglu_t",
        "qwen35_ane_affine_swiglu_t",
        "qwen35_ane_dual_q4_swiglu_t",
        "qwen35_ane_dual_affine_swiglu_t",
    ):
        decl = header.split(f"mlx::core::array {name}(", 1)[1].split(");", 1)[0]
        assert "bool geglu = false" in decl
        block = next(
            part for part in bindings.split("  m.def(") if f'"{name}",' in part
        )
        assert '"geglu"_a = false' in block
    # No GeGLU-specific entry points were added.
    assert "geglu_t" not in header
    assert "qwen35_ane_fused_geglu_available" in header


def test_geglu_capability_probe_is_bound():
    bindings = (_CSRC / "bindings.cpp").read_text(encoding="utf-8")
    assert '"qwen35_ane_fused_geglu_available"' in bindings


def test_ane_compile_bindings_still_release_the_python_gil():
    bindings = (_CSRC / "bindings.cpp").read_text(encoding="utf-8")
    blocks = bindings.split("  m.def(")
    guard = "nb::call_guard<nb::gil_scoped_release>()"
    for name in (
        "qwen35_ane_compile_linear",
        "qwen35_ane_compile_linear_bank",
    ):
        block = next(part for part in blocks if f'"{name}"' in part)
        assert guard in block


# --- Shared runtime --------------------------------------------------------


def _config(**kwargs) -> ane_patch._AnePrefillConfig:
    base = ane_patch._AnePrefillConfig(
        sequence_length=2048, fraction=0.5, variant=8
    )
    return replace(base, **kwargs) if kwargs else base


def test_config_defaults_to_swiglu():
    assert _config().geglu is False
    assert ane_patch._glu(_config()) is ane_patch.swiglu


def test_geglu_config_selects_the_gemma_activation():
    from mlx_lm.models.gemma4_text import geglu

    assert ane_patch._glu(_config(geglu=True)) is geglu


def test_glu_matches_the_reference_activation_numerically():
    gate = mx.array([[-4.0, -0.25, 0.0, 0.25, 3.0]], dtype=mx.float32)
    up = mx.ones_like(gate)
    got = ane_patch._glu(_config(geglu=True))(gate, up)
    expected = nn.gelu_approx(gate) * up
    assert mx.allclose(got, expected, atol=1e-6).item()


def test_geglu_rejects_cpu_sharing_and_fused_down():
    """Those merges are SwiGLU-only; a silent wrong activation is worse."""
    model = SimpleNamespace()
    for kwargs in (
        {"cpu_fraction": 0.1},
        {"cpu_down_fraction": 0.1},
        {"fused_down": True},
    ):
        with pytest.raises(ValueError, match="GeGLU"):
            ane_patch.enable_qwen35_ane_prefill(model, geglu=True, **kwargs)


def test_install_dispatch_wraps_the_named_classes(monkeypatch):
    monkeypatch.setattr(ane_patch, "_PATCHED_CLASSES", set())
    wrapped = []
    monkeypatch.setattr(ane_patch, "_wrap_class", wrapped.append)

    class _A:
        pass

    class _B:
        pass

    assert ane_patch._install_dispatch((_A, _B)) is True
    assert wrapped == [_A, _B]


# --- Gemma 4 eligibility ---------------------------------------------------


class _Gemma4MLP(nn.Module):
    """The 31B shape scaled down: intermediate = 4x hidden, no bias."""

    def __init__(self, bits=4, group_size=64, hidden=128, intermediate=512):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(
            hidden, intermediate, bias=False, group_size=group_size, bits=bits
        )
        self.up_proj = nn.QuantizedLinear(
            hidden, intermediate, bias=False, group_size=group_size, bits=bits
        )
        self.down_proj = nn.QuantizedLinear(
            intermediate, hidden, bias=False, group_size=group_size, bits=bits
        )
        # The ANE path requires half-precision scales, as a real checkpoint has.
        for linear in (self.gate_proj, self.up_proj, self.down_proj):
            linear.scales = linear.scales.astype(mx.float16)
            linear.biases = linear.biases.astype(mx.float16)


class _Gemma4MoEMLP(nn.Module):
    """A routed-expert block: no gate/up/down triple to wrap."""

    def __init__(self):
        super().__init__()
        self.router = nn.QuantizedLinear(128, 8, bias=False, group_size=64, bits=4)


@pytest.mark.parametrize("bits", (4, 5, 6, 8))
@pytest.mark.parametrize("group_size", (64, 128))
def test_supported_quantizations_are_eligible(bits, group_size):
    assert ane_patch._eligible_pair(_Gemma4MLP(bits=bits, group_size=group_size))


def test_three_bit_gate_up_is_rejected():
    """No q3 GPU suffix exists to pair with the ANE prefix."""
    mlp = _Gemma4MLP(bits=4)
    mlp.gate_proj.bits = 3
    assert not ane_patch._eligible_pair(mlp)


def test_double_wide_mlp_is_eligible():
    """E-series widths are per-procedure, so the bank derives them itself."""
    assert ane_patch._eligible_pair(_Gemma4MLP(intermediate=1024))


def test_moe_expert_block_is_not_a_dispatch_target():
    assert not ane_patch._eligible_pair(_Gemma4MoEMLP())


# --- Enable path -----------------------------------------------------------


def test_dispatch_targets_are_the_text_and_multimodal_mlps():
    import mlx_lm.models.gemma4_text as gemma4_text

    classes = gemma4_patch._mlp_classes()
    assert gemma4_text.MLP in classes
    assert all(isinstance(cls, type) for cls in classes)


def test_enable_delegates_with_geglu_and_the_gemma_classes(monkeypatch):
    seen = {}

    def _fake_enable(model, **kwargs):
        seen.update(kwargs)
        return 60

    monkeypatch.setattr(gemma4_patch, "enable_qwen35_ane_prefill", _fake_enable)
    count = gemma4_patch.enable_gemma4_ane_prefill(
        SimpleNamespace(), sequence_length=2048, fraction=0.4, max_layers=60
    )
    assert count == 60
    assert seen["geglu"] is True
    assert seen["gdn"] is False
    assert seen["fraction"] == 0.4
    assert seen["mlp_classes"] == gemma4_patch._mlp_classes()


def test_moe_checkpoints_are_refused_without_raising(monkeypatch):
    monkeypatch.setattr(
        gemma4_patch,
        "enable_qwen35_ane_prefill",
        lambda *a, **k: pytest.fail("MoE must not reach the shared runtime"),
    )
    model = SimpleNamespace(args=SimpleNamespace(enable_moe_block=True))
    assert gemma4_patch.enable_gemma4_ane_prefill(model) == 0


def test_moe_is_seen_through_the_multimodal_wrapper(monkeypatch):
    """A text-only 26B A4B checkpoint still loads as ``gemma4.Model``.

    Its top-level ``ModelArgs`` carries no ``enable_moe_block`` -- the flag is
    on the nested ``language_model.args`` -- so a gate that reads only the
    outermost holder accelerates 30 MoE MLPs it was written to refuse.
    """
    monkeypatch.setattr(
        gemma4_patch,
        "enable_qwen35_ane_prefill",
        lambda *a, **k: pytest.fail("MoE must not reach the shared runtime"),
    )
    model = SimpleNamespace(
        args=SimpleNamespace(),  # gemma4.ModelArgs has no enable_moe_block
        language_model=SimpleNamespace(
            args=SimpleNamespace(enable_moe_block=True)
        ),
    )
    assert gemma4_patch._is_moe(model) is True
    assert gemma4_patch.enable_gemma4_ane_prefill(model) == 0


def test_kill_switch_disables_the_gemma_path(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN35_ANE_PREFILL", "0")
    monkeypatch.setattr(
        gemma4_patch,
        "enable_qwen35_ane_prefill",
        lambda *a, **k: pytest.fail("kill switch must short-circuit"),
    )
    assert gemma4_patch.enable_gemma4_ane_prefill(SimpleNamespace()) == 0


def test_missing_mlp_classes_return_zero(monkeypatch):
    monkeypatch.setattr(gemma4_patch, "_mlp_classes", tuple)
    assert gemma4_patch.enable_gemma4_ane_prefill(SimpleNamespace()) == 0


def test_status_and_release_delegate_to_the_shared_runtime(monkeypatch):
    monkeypatch.setattr(
        gemma4_patch, "qwen35_ane_prefill_status", lambda model: {"mlp_layers": 7}
    )
    monkeypatch.setattr(
        gemma4_patch, "release_qwen35_ane_prefill", lambda model: (7, 14)
    )
    model = SimpleNamespace()
    assert gemma4_patch.gemma4_ane_prefill_status(model)["mlp_layers"] == 7
    assert gemma4_patch.release_gemma4_ane_prefill(model) == (7, 14)


def test_variant_table_records_a_reason_for_every_unsupported_variant():
    rows = gemma4_patch.VARIANT_SUPPORT
    assert any(row["supported"] for row in rows)
    for row in rows:
        assert row["variant"] and row["model_type"]
        if not row["supported"]:
            assert row["reason"]


def test_the_driver_count_decides_not_the_chip_name(monkeypatch):
    """The device is the authority; the marketing name only describes the SKU.

    Both directions matter. A two-die part the name parser does not recognise
    as Ultra would otherwise lose the dual path, and a machine reporting one
    engine must not get two banks because its model number says Ultra.
    """
    import omlx.utils.hardware as hardware

    for measured, chip, expected in (
        (2, "Apple M1 Max", True),
        (1, "Apple M3 Ultra", False),
        (2, "Apple M3 Ultra", True),
        (1, "Apple M1 Max", False),
    ):
        monkeypatch.setattr(
            hardware, "get_ane_instance_count", lambda m=measured: m
        )
        monkeypatch.setattr(hardware, "get_chip_name", lambda c=chip: c)
        assert gemma4_patch._has_two_ane_dies() is expected, (measured, chip)


def test_chip_name_decides_when_the_property_is_absent(monkeypatch):
    """A host without the driver property keeps the previous behaviour."""
    import omlx.utils.hardware as hardware

    monkeypatch.setattr(hardware, "get_ane_instance_count", lambda: None)
    for chip, expected in (
        ("Apple M3 Ultra", True),
        ("Apple M1 Max", False),
        ("Apple M4 Pro", False),
        ("Apple M2", False),
    ):
        monkeypatch.setattr(hardware, "get_chip_name", lambda c=chip: c)
        assert gemma4_patch._has_two_ane_dies() is expected, chip


def test_undetectable_chip_assumes_one_ane(monkeypatch):
    """A detection failure must not leave two banks requested.

    Both sources have to fail for this to be the fallback under test -- with
    the driver property readable it answers first and the chip name is never
    consulted.
    """
    import omlx.utils.hardware as hardware

    def boom():
        raise OSError("no chip info")

    monkeypatch.setattr(hardware, "get_ane_instance_count", lambda: None)
    monkeypatch.setattr(hardware, "get_chip_name", boom)
    assert gemma4_patch._has_two_ane_dies() is False


def test_dual_ane_downgrades_on_single_ane_hardware(monkeypatch):
    """dual_ane reads as "allow dual", not "pretend there are two"."""
    captured = {}

    def fake_enable(model, **kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(gemma4_patch, "enable_qwen35_ane_prefill", fake_enable)
    monkeypatch.setattr(gemma4_patch, "_mlp_classes", lambda: (object,))
    monkeypatch.setattr(gemma4_patch, "_has_two_ane_dies", lambda: False)

    gemma4_patch.enable_gemma4_ane_prefill(SimpleNamespace(), dual_ane=True)
    assert captured["dual_ane"] is False


def test_dual_ane_survives_on_two_die_hardware(monkeypatch):
    """The downgrade must not fire where the second die genuinely exists."""
    captured = {}

    def fake_enable(model, **kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(gemma4_patch, "enable_qwen35_ane_prefill", fake_enable)
    monkeypatch.setattr(gemma4_patch, "_mlp_classes", lambda: (object,))
    monkeypatch.setattr(gemma4_patch, "_has_two_ane_dies", lambda: True)

    gemma4_patch.enable_gemma4_ane_prefill(SimpleNamespace(), dual_ane=True)
    assert captured["dual_ane"] is True


def test_ane_prefill_backend_names_the_gemma_family():
    """The admin UI gates on this rather than matching model types itself."""
    from omlx.model_settings import ane_prefill_backend

    for model_type in ("gemma4", "gemma4_text", "gemma4-unified"):
        assert ane_prefill_backend(model_type) == "gemma4"
    assert ane_prefill_backend("qwen3_5") == "qwen"
