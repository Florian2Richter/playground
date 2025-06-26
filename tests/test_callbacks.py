from importlib import import_module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_worker_callback_calls():
    mod = import_module('scripts.callback')

    class Dummy(mod.callback):
        def __init__(self):
            self.begin_called = 0
            self.iter_called = 0

        def on_begin(self):
            self.begin_called += 1

        def on_iteration(self):
            self.iter_called += 1

    dummy = Dummy()
    mod.worker([dummy])
    assert dummy.begin_called == 1
    assert dummy.iter_called == 50


def test_worker_playground_callback_calls():
    mod = import_module('scripts.callback_playground')

    class Dummy(mod.callback):
        def __init__(self):
            self.begin_called = 0
            self.iter_called = 0

        def on_begin(self):
            self.begin_called += 1

        def on_iteration(self):
            self.iter_called += 1

    dummy = Dummy()
    mod.worker([dummy])
    assert dummy.begin_called == 1
    assert dummy.iter_called == 50


def test_worker_callback2_calls():
    mod = import_module('scripts.callback2')

    class Dummy(mod.Callback):
        def __init__(self):
            self.begin_called = 0
            self.iter_called = 0

        def on_begin(self):
            self.begin_called += 1

        def on_iteration(self):
            self.iter_called += 1

    dummy = Dummy()
    mod.worker([dummy])
    assert dummy.begin_called == 1
    assert dummy.iter_called == 50
