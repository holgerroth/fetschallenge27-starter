"""Loading and validation helpers for participant local training."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

try:
    from nvflare.app_common.abstract.fl_model import FLModel
except ModuleNotFoundError:  # pragma: no cover - used only without NVFLARE
    from .compat import FLModel


def load_participant_local_train(module_path: Path) -> Callable:
    """Load the participant-defined ``local_train`` function."""
    spec = importlib.util.spec_from_file_location("participant.client", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load participant client module from {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    local_train = getattr(module, "local_train", None)
    if not callable(local_train):
        raise AttributeError(
            "participant/client.py must expose callable local_train()."
        )
    return local_train


def parameter_spec(params: Mapping) -> dict[str, tuple[tuple[int, ...], str]]:
    """Return the shape and dtype expected for each model parameter."""
    return {
        key: (tuple(int(value) for value in param.shape), str(param.dtype))
        for key, param in params.items()
    }


def validate_local_train_result(
    result: FLModel,
    expected_spec: Mapping[str, tuple[tuple[int, ...], str]],
) -> FLModel:
    """Validate that ``local_train`` returned a compatible full model."""
    if not isinstance(result, FLModel):
        raise TypeError("local_train() must return an FLModel.")
    if not isinstance(result.params, Mapping) or not result.params:
        raise ValueError(
            "local_train() must return FLModel.params as a non-empty mapping."
        )

    params_type = getattr(result.params_type, "value", result.params_type)
    if params_type not in (None, "FULL"):
        raise ValueError(
            "local_train() must return full parameters; params_type must be FULL."
        )

    expected_keys = set(expected_spec)
    returned_keys = set(result.params)
    if returned_keys != expected_keys:
        missing = sorted(expected_keys - returned_keys)
        unexpected = sorted(returned_keys - expected_keys)
        raise ValueError(
            "local_train() returned incompatible parameter keys: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for key, value in result.params.items():
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(
                f"Parameter {key!r} must be a tensor or array with shape and dtype."
            )
        returned = (tuple(int(size) for size in value.shape), str(value.dtype))
        if returned != expected_spec[key]:
            raise ValueError(
                f"Parameter {key!r} has shape/dtype {returned}; expected {expected_spec[key]}."
            )

    if result.meta is not None and not isinstance(result.meta, dict):
        raise TypeError("FLModel.meta returned by local_train() must be a dictionary.")
    if result.metrics is not None and not isinstance(result.metrics, dict):
        raise TypeError(
            "FLModel.metrics returned by local_train() must be a dictionary."
        )
    return result
