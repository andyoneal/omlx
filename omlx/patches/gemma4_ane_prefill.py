# SPDX-License-Identifier: Apache-2.0
"""Opt-in ANE/GPU hybrid prefill for dense Gemma 4 MLPs.

Splits each ``gate_proj``/``up_proj`` pair between an INT8 ANE channel prefix
and the quantized GPU suffix, merging the two while applying GeGLU so the full
fused gate/up result is never materialized. The bank ladder, fixed-shape
eligibility, memory admission and teardown are shared with the Qwen
implementation; only the activation and the dispatch targets differ. The down
projection, CPU sharing and the MoE experts keep the existing GPU path.
Enabled through the per-model ``gemma4_ane_prefill_*`` settings.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from omlx.patches.qwen35_ane_prefill import (
    enable_qwen35_ane_prefill,
    qwen35_ane_prefill_status,
    release_qwen35_ane_prefill,
)

logger = logging.getLogger(__name__)

# Dense text and multimodal MLPs share ``__call__(self, x)``, which the shared
# wrapper's ``patched(self, x, *args, **kwargs)`` already accepts. Resolved
# leniently: a checkout without mlx-vlm still accelerates the text path.
#
# The 12B unified checkpoint needs no entry of its own: its model does
# ``from ..gemma4.language import LanguageModel``, so its decoder layers carry
# the second class here already.
_MLP_MODULES = (
    ("mlx_lm.models.gemma4_text", "MLP"),
    ("mlx_vlm.models.gemma4.language", "MLP"),
)

# Which Gemma 4 variants this path accelerates, and what blocks the rest.
# Data rather than stub functions, so extending it is a scope decision rather
# than a code-shape decision.
VARIANT_SUPPORT: tuple[dict[str, Any], ...] = (
    {
        "variant": "31B dense",
        "model_type": "gemma4 / gemma4_text",
        "supported": True,
        "reason": "",
    },
    {
        "variant": "26B-A4B MoE",
        "model_type": "gemma4 (enable_moe_block)",
        "supported": False,
        "reason": (
            "Routed experts run through SwitchGLU. The dense shared mlp "
            "survives beside them, so a gate change plus width bookkeeping "
            "would reach it -- but that alone measures 0.879x at best legal "
            "width, because the block displaces ~1.35 ms of GPU work against "
            "a ~1.5 ms fixed dispatch cost. What pays on this model is the "
            "dense subset, attention projections plus the dense mlp at 53.4% "
            "of projection FLOPs, and no attention-projection path exists yet."
        ),
    },
    {
        "variant": "12B unified",
        "model_type": "gemma4_unified",
        "supported": True,
        "reason": "",
    },
    {
        "variant": "E4B / E2B",
        "model_type": "gemma4",
        "supported": False,
        "reason": (
            "use_double_wide_mlp and num_kv_shared_layers give two MLP widths "
            "in one model. The procedure bank already derives each shape "
            "independently, so this is prep-side work only."
        ),
    },
)


def _mlp_classes() -> tuple[type, ...]:
    classes: list[type] = []
    for module_name, attr in _MLP_MODULES:
        try:
            cls = getattr(importlib.import_module(module_name), attr, None)
        except Exception:
            logger.debug("%s unavailable for Gemma 4 ANE dispatch", module_name)
            continue
        if isinstance(cls, type):
            classes.append(cls)
    return tuple(classes)


def _is_moe(model: Any) -> bool:
    """Whether this checkpoint routes through MoE experts.

    ``enable_moe_block`` lives on the *text* config. A Gemma 4 checkpoint whose
    top-level ``model_type`` is ``gemma4`` loads as the multimodal wrapper even
    when its weights are text-only, and that wrapper's own ``ModelArgs`` has no
    such field -- so checking only the outermost holder reads False on a MoE
    model and the gate below never fires. Walk the language-model holder too.
    """
    holders = [model, getattr(model, "language_model", None)]
    for owner in holders:
        if owner is None:
            continue
        for holder in (getattr(owner, "args", None), getattr(owner, "config", None)):
            if getattr(holder, "enable_moe_block", False):
                return True
    return False


def _has_two_ane_dies() -> bool:
    """Whether this machine exposes two physical ANE instances.

    Defers to the shared runtime, which applies the same fallback itself. Kept
    as a name here because the log line below is family-specific.
    """
    from omlx.patches.qwen35_ane_prefill import ane_instance_count

    return ane_instance_count() >= 2


def enable_gemma4_ane_prefill(
    model: Any,
    *,
    sequence_length: int = 128,
    fraction: float = 0.50,
    max_layers: int = 60,
    dual_ane: bool = True,
    tail_padding_min_tokens: int = 0,
) -> int:
    """Enable the hybrid ANE backend on eligible dense Gemma 4 MLPs.

    Returns the number of accelerated MLP modules; zero is a safe no-op for
    other families, MoE checkpoints and unsupported runtimes.
    """
    env = os.environ.get("OMLX_QWEN35_ANE_PREFILL", "").strip().lower()
    if env in ("0", "false", "off"):
        # One kill switch for the shared runtime, as DeepSeek does.
        logger.info("Gemma 4 ANE prefill disabled by OMLX_QWEN35_ANE_PREFILL")
        return 0
    if _is_moe(model):
        logger.warning(
            "Gemma 4 ANE prefill does not support MoE checkpoints; skipped"
        )
        return 0
    classes = _mlp_classes()
    if not classes:
        logger.warning("Gemma 4 MLP class unavailable; ANE prefill skipped")
        return 0
    if dual_ane and not _has_two_ane_dies():
        # Two banks on one die doubles bank memory and puts two submitting
        # threads on the same device. Treat the setting as "allow dual".
        logger.info(
            "Gemma 4 ANE prefill: one physical ANE detected, compiling one "
            "program per layer instead of the requested dual-ANE split"
        )
        dual_ane = False
    return enable_qwen35_ane_prefill(
        model,
        sequence_length=sequence_length,
        fraction=fraction,
        max_layers=max_layers,
        dual_ane=dual_ane,
        tail_padding_min_tokens=tail_padding_min_tokens,
        gdn=False,
        geglu=True,
        mlp_classes=classes,
    )


def gemma4_ane_prefill_status(model: Any) -> dict:
    return qwen35_ane_prefill_status(model)


def release_gemma4_ane_prefill(model: Any) -> tuple[int, int]:
    return release_qwen35_ane_prefill(model)
