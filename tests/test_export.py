import torch


def test_torch_export_dynamic_sequence(tiny_model):
    from export import InferenceWrapper, export_graph
    module = InferenceWrapper(tiny_model.eval())
    program = export_graph(module, torch.ones((1, 8), dtype=torch.long), "none")
    assert program.module()(input_ids=torch.ones((1, 11), dtype=torch.long)).shape == (1, 10)
