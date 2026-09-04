from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fets27_challenge.local_training import (
    FLModel,
    load_participant_local_train,
    parameter_spec,
    validate_local_train_result,
)
from fets27_challenge.runtime import resolve_client_scripts


def test_load_participant_local_train():
    local_train = load_participant_local_train(Path("participant/client.py"))
    assert callable(local_train)


def test_validate_local_train_result_accepts_compatible_full_model():
    expected = parameter_spec({"weight": np.zeros((2, 3), dtype=np.float32)})
    result = FLModel(
        params={"weight": np.ones((2, 3), dtype=np.float32)},
        params_type="FULL",
        meta={"participant_stat": 1.0},
    )

    assert validate_local_train_result(result, expected) is result


@pytest.mark.parametrize(
    "params",
    [
        {"different": np.ones((2, 3), dtype=np.float32)},
        {"weight": np.ones((3, 2), dtype=np.float32)},
        {"weight": np.ones((2, 3), dtype=np.float64)},
    ],
)
def test_validate_local_train_result_rejects_incompatible_params(params):
    expected = parameter_spec({"weight": np.zeros((2, 3), dtype=np.float32)})
    with pytest.raises(ValueError, match="incompatible|shape/dtype"):
        validate_local_train_result(
            FLModel(params=params, params_type="FULL"), expected
        )


def test_validate_local_train_result_requires_flmodel():
    with pytest.raises(TypeError, match="must return an FLModel"):
        validate_local_train_result({}, {})


def test_resolve_client_scripts_returns_locked_and_participant_files():
    locked_client, participant_client = resolve_client_scripts(Path.cwd())
    assert locked_client == (Path.cwd() / "src/fets27_challenge/client.py").resolve()
    assert participant_client == (Path.cwd() / "participant/client.py").resolve()
