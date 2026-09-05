from __future__ import annotations

from types import SimpleNamespace

import pytest

from fets27_challenge import data_pipeline


class _Dataset:
    def __init__(self, *, data, transform):
        self.data = data
        self.transform = transform


class _DataLoader:
    calls = []

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs
        self.calls.append(kwargs)


def _patch_lightweight_runtime(monkeypatch):
    _DataLoader.calls = []
    monkeypatch.setattr(data_pipeline, "require_runtime_dependencies", lambda: None)
    monkeypatch.setattr(
        data_pipeline,
        "load_decathlon_datalist",
        lambda **kwargs: [kwargs["data_list_key"]],
    )
    monkeypatch.setattr(data_pipeline, "_build_train_transform", lambda *_: "train")
    monkeypatch.setattr(data_pipeline, "_build_valid_transform", lambda *_: "valid")
    monkeypatch.setattr(data_pipeline, "Dataset", _Dataset)
    monkeypatch.setattr(data_pipeline, "DataLoader", _DataLoader)
    monkeypatch.setattr(
        data_pipeline,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(data_pipeline, "SlidingWindowInferer", lambda **kwargs: kwargs)
    monkeypatch.setattr(data_pipeline, "Activations", lambda **kwargs: kwargs)
    monkeypatch.setattr(data_pipeline, "AsDiscrete", lambda **kwargs: kwargs)
    monkeypatch.setattr(data_pipeline, "Compose", lambda transforms: transforms)
    monkeypatch.setattr(data_pipeline, "DiceMetric", lambda **kwargs: kwargs)


def _build(data_loader_workers=2):
    return data_pipeline.build_dataloaders(
        dataset_base_dir="dataset",
        datalist_json_path="site.json",
        label_transform="brats_multi_channel",
        batch_size=2,
        cache_rate=0.0,
        roi_size=(128, 128, 128),
        infer_roi_size=(32, 32, 32),
        data_loader_workers=data_loader_workers,
    )


def test_build_dataloaders_uses_safe_default(monkeypatch):
    _patch_lightweight_runtime(monkeypatch)

    data_pipeline.build_dataloaders(
        dataset_base_dir="dataset",
        datalist_json_path="site.json",
        label_transform="brats_multi_channel",
        batch_size=2,
        cache_rate=0.0,
        roi_size=(128, 128, 128),
        infer_roi_size=(32, 32, 32),
    )

    assert len(_DataLoader.calls) == 2
    assert all(call["num_workers"] == 2 for call in _DataLoader.calls)
    assert all(call["persistent_workers"] is True for call in _DataLoader.calls)
    assert all(call["prefetch_factor"] == 4 for call in _DataLoader.calls)


def test_build_dataloaders_supports_synchronous_loading(monkeypatch):
    _patch_lightweight_runtime(monkeypatch)

    _build(data_loader_workers=0)

    assert len(_DataLoader.calls) == 2
    assert all(call["num_workers"] == 0 for call in _DataLoader.calls)
    assert all("persistent_workers" not in call for call in _DataLoader.calls)
    assert all("prefetch_factor" not in call for call in _DataLoader.calls)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_build_dataloaders_rejects_invalid_worker_count(monkeypatch, value):
    _patch_lightweight_runtime(monkeypatch)

    with pytest.raises(ValueError, match="non-negative integer"):
        _build(data_loader_workers=value)
