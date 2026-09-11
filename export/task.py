"""Hydra task for exporting the segmenter to ExecuTorch."""

from pathlib import Path

import torch
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.export import Dim

from model.model import PinyinSegmentModel


class InferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, return_dict=False)[0]


DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def export_graph(module, example, quantization, sequence_min=3, sequence_max=512, strict=True,
                 per_channel=True, dynamic=True):
    dynamic_shapes = {
        "input_ids": {
            # Mobile inference is one query at a time. PyTorch export also
            # specializes a size-1 example batch, so keep that dimension
            # static and make only the sequence length dynamic.
            1: Dim("sequence", min=sequence_min, max=sequence_max),
        }
    }
    if quantization == "none":
        return torch.export.export(
            module, (), {"input_ids": example}, dynamic_shapes=dynamic_shapes, strict=strict
        )
    if quantization != "w8a8":
        raise ValueError("quantization must be 'none' or 'w8a8'")

    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    # Quantize an already-dynamic exported module. Capturing a static graph
    # here would specialize PT2E observers to the calibration length.
    captured = torch.export.export(
        module, (), {"input_ids": example}, dynamic_shapes=dynamic_shapes
    ).module()
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(
        is_per_channel=per_channel, is_dynamic=dynamic
    ))
    prepared = prepare_pt2e(captured, quantizer)
    with torch.no_grad():
        prepared(input_ids=example)
    converted = convert_pt2e(prepared)
    return torch.export.export(
        converted, (), {"input_ids": example}, dynamic_shapes=dynamic_shapes, strict=strict
    )


def write_manifest(pte_path, output_path, quantization, model_version, format_version, layout):
    from executorch.codegen.tools.gen_oplist import _get_kernel_metadata_for_model, _get_operators
    from torchgen.selective_build.operator import SelectiveBuildOperator

    operators = {}
    for name in sorted(set(_get_operators(pte_path))):
        operators[name] = SelectiveBuildOperator.from_yaml_dict(name, {
            "is_root_operator": True,
            "is_used_for_training": False,
            "include_all_overloads": False,
            "debug_info": ["phono-pinyin-segment"],
        }).to_dict()
    manifest = {
        "model": {"version": model_version, "format_version": format_version,
                  "quantization": quantization, "layout": layout},
        "operators": operators,
        "custom_classes": [], "build_features": [],
        "include_all_non_op_selectives": False, "include_all_operators": False,
        "kernel_metadata": {},
        "et_kernel_metadata": _get_kernel_metadata_for_model(pte_path),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def _absolute_path(path):
    return Path(to_absolute_path(str(Path(path).expanduser())))


def run_export(task_cfg: DictConfig):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    device = torch.device(str(task_cfg.device))
    if device.type != "cpu":
        raise ValueError("ExecuTorch XNNPACK export currently requires task.device=cpu")
    try:
        dtype = DTYPES[str(task_cfg.dtype).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported export dtype: {task_cfg.dtype}") from exc

    example_cfg = task_cfg.example_inputs
    sequence_length = int(example_cfg.sequence_length)
    sequence_min = int(example_cfg.sequence_min)
    sequence_max = int(example_cfg.sequence_max)
    if not 3 <= sequence_min <= sequence_length <= sequence_max:
        raise ValueError("sequence lengths must satisfy 3 <= min <= example <= max")

    checkpoint = _absolute_path(task_cfg.checkpoint_dir)
    output_dir = _absolute_path(task_cfg.output_dir)
    model = PinyinSegmentModel.from_pretrained(checkpoint).to(device=device, dtype=dtype).eval()
    model.to(memory_format=torch.channels_last)
    wrapper = InferenceWrapper(model).eval()
    example = torch.ones((1, sequence_length), dtype=torch.long, device=device)
    exported = export_graph(
        wrapper, example, str(task_cfg.quantization.mode), sequence_min, sequence_max,
        bool(task_cfg.export.strict), bool(task_cfg.quantization.per_channel),
        bool(task_cfg.quantization.dynamic),
    )
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    program = edge.to_executorch()

    output_dir.mkdir(parents=True, exist_ok=True)
    pte_path = output_dir / str(task_cfg.outputs.model)
    with pte_path.open("wb") as f:
        program.write_to_file(f)
    if bool(task_cfg.manifests.enabled):
        write_manifest(
            pte_path, output_dir / str(task_cfg.outputs.manifest),
            str(task_cfg.quantization.mode), str(task_cfg.model_metadata.version),
            str(task_cfg.model_metadata.format_version), str(task_cfg.model_metadata.layout),
        )
    print(f"Saved {pte_path}")
