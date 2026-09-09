"""Export the segmenter to ExecuTorch with optional dynamic W8A8 quantization."""

import argparse
import os

import torch
import yaml
from torch.export import Dim

from model.model import PinyinSegmentModel


class InferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, return_dict=False)[0]


def export_graph(module, example, quantization):
    dynamic_shapes = {
        "input_ids": {
            # Mobile inference is one query at a time. PyTorch export also
            # specializes a size-1 example batch, so keep that dimension
            # static and make only the sequence length dynamic.
            1: Dim("sequence", min=3, max=512),
        }
    }
    if quantization == "none":
        return torch.export.export(
            module, (), {"input_ids": example}, dynamic_shapes=dynamic_shapes
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
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))
    prepared = prepare_pt2e(captured, quantizer)
    with torch.no_grad():
        prepared(input_ids=example)
    converted = convert_pt2e(prepared)
    return torch.export.export(
        converted, (), {"input_ids": example}, dynamic_shapes=dynamic_shapes
    )


def write_manifest(pte_path, output_path, quantization):
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
        "model": {"format_version": "1.0", "quantization": quantization, "layout": "BHWC"},
        "operators": operators,
        "custom_classes": [], "build_features": [],
        "include_all_non_op_selectives": False, "include_all_operators": False,
        "kernel_metadata": {},
        "et_kernel_metadata": _get_kernel_metadata_for_model(pte_path),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints/v1_0-small-alpha02/final_model")
    parser.add_argument("--output-dir", default="./export_output")
    parser.add_argument("--quantization", choices=("none", "w8a8"), default="w8a8")
    args = parser.parse_args()

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    model = PinyinSegmentModel.from_pretrained(args.checkpoint).eval()
    model.to(memory_format=torch.channels_last)
    wrapper = InferenceWrapper(model).eval()
    example = torch.ones((1, 16), dtype=torch.long)
    exported = export_graph(wrapper, example, args.quantization)
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    program = edge.to_executorch()

    os.makedirs(args.output_dir, exist_ok=True)
    pte_path = os.path.join(args.output_dir, "pinyin_segment.pte")
    with open(pte_path, "wb") as f:
        program.write_to_file(f)
    write_manifest(
        pte_path, os.path.join(args.output_dir, "pinyin_segment_ops.yaml"), args.quantization
    )
    print(f"Saved {pte_path}")


if __name__ == "__main__":
    main()
