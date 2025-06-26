from importlib import import_module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_flo_generator_yields_in_order():
    mod = import_module('scripts.generator_example')
    gen = mod.flo_generator([1, 2, 3])
    assert list(gen) == [1, 2, 3]
