from __future__ import annotations

from fets27_challenge.client import _PreloadedDataLoader


class _RecordingLoader:
    def __init__(self):
        self.iterations = 0
        self.dataset = [1, 2]

    def __iter__(self):
        self.iterations += 1
        return iter([self.iterations])

    def __len__(self):
        return 1


def test_preloaded_data_loader_preserves_prefetched_iterator():
    loader = _RecordingLoader()
    wrapped = _PreloadedDataLoader(loader)

    wrapped.preload()

    assert loader.iterations == 1
    assert list(wrapped) == [1]
    assert loader.iterations == 1
    assert list(wrapped) == [2]
    assert loader.iterations == 2
    assert len(wrapped) == 1
    assert wrapped.dataset == [1, 2]
