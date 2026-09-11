import os

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    os.environ["WANDB_CONSOLE"] = "off"
    if cfg.system.expandable_segments_enabled:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    if cfg.system.tf32_enabled:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if cfg.system.set_seed:
        torch.manual_seed(cfg.system.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.system.seed)
    torch.set_float32_matmul_precision("high")

    task_name = cfg.task.get("task_type", "train")
    if task_name == "export":
        from export import run_export

        run_export(cfg.task)
    elif task_name == "train":
        from tasks.train import Trainer

        Trainer(cfg).train()
    else:
        raise ValueError(f"Unsupported task_type: {task_name}")


if __name__ == "__main__":
    main()
