import pytest
import torch

from demo import build_parser, resolve_runtime


def test_demo_requires_model_and_legal_vocabulary():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nihaoma"])


def test_demo_cpu_auto_dtype():
    device, dtype = resolve_runtime("cpu", "auto")
    assert device == torch.device("cpu")
    assert dtype == torch.float32
