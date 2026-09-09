import logging
import os

import torch
import wandb
from omegaconf import OmegaConf
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from streaming import StreamingDataLoader
from torch.utils.data import DataLoader
from transformers import get_cosine_with_min_lr_schedule_with_warmup

from datasets_pipeline import PinyinSegmentStreamingDataset, create_dataset, make_collate_fn, transform_batch
from metrics import SegmentationMetrics
from model.config import build_config_from_dict
from model.model import PinyinSegmentModel
from tokenizer import PinyinCharTokenizer


def _progress(cfg):
    pcfg = cfg.output.progress_bar
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=pcfg.get("bar_width")), TaskProgressColumn(),
        TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn(),
        TextColumn("{task.fields[postfix]}"),
        refresh_per_second=pcfg.get("refresh_per_second", 10),
        transient=pcfg.get("transient", True),
    )


def _optimizer_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 and "embed" not in name else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.system.device)
        self.use_amp = cfg.system.mixed_precision == "bf16" and self.device.type == "cuda"
        self.loss_kwargs = {
            "loss_type": cfg.task.loss_type,
            "focal_alpha": cfg.task.focal_loss_alpha,
            "focal_gamma": cfg.task.focal_loss_gamma,
        }
        logging.basicConfig(
            level=getattr(logging, cfg.output.logging.log_level.upper(), logging.WARNING),
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        self.tokenizer = PinyinCharTokenizer.from_config(cfg.dataset.vocabs_config)
        model_cfg = build_config_from_dict(
            OmegaConf.to_container(cfg.model, resolve=True),
            self.tokenizer.vocab_size, self.tokenizer.pad_token_id,
        )
        self.model = PinyinSegmentModel(model_cfg).to(self.device)
        self.model.to(memory_format=torch.channels_last)

        if cfg.task.get("load_from"):
            self.model = PinyinSegmentModel.from_pretrained(cfg.task.load_from).to(self.device)
            self.model.to(memory_format=torch.channels_last)
        if cfg.system.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.log_cfg = cfg.logging.train
        self.checkpoint_dir = self.log_cfg.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.log_cfg.log_with_wandb:
            wandb.init(
                project=self.log_cfg.project_name, name=self.log_cfg.run_name,
                notes=self.log_cfg.get("notes", ""),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            wandb.watch(self.model, log="all", log_freq=self.log_cfg.histogram_interval)

        augmentation = OmegaConf.to_container(cfg.dataset.augmentation, resolve=True)
        self.train_ds = PinyinSegmentStreamingDataset(
            local=cfg.dataset.train_dir_mds,
            tokenizer=self.tokenizer, augmentation=augmentation,
            shuffle=True, shuffle_algo=cfg.dataset.shuffle.algo,
            shuffle_block_size=cfg.dataset.shuffle.blocksize,
            cache_limit=cfg.dataset.shuffle.cache_limit,
            batch_size=cfg.task.batchsize,
        )
        collate = make_collate_fn(self.tokenizer.pad_token_id)
        self.train_loader = StreamingDataLoader(
            self.train_ds, batch_size=cfg.task.batchsize,
            num_workers=cfg.system.num_workers, collate_fn=collate,
        )
        self.val_ds = create_dataset(cfg.dataset.val_dir, cfg.system.keep_in_memory)
        self.val_ds.set_transform(lambda batch: transform_batch(batch, self.tokenizer, {}))
        self.val_loader = DataLoader(
            self.val_ds, batch_size=cfg.task.val_batchsize, shuffle=False,
            num_workers=cfg.system.num_workers, pin_memory=self.device.type == "cuda",
            collate_fn=collate,
        )

        groups = _optimizer_groups(self.model, cfg.task.weight_decay)
        if cfg.system.optim_8bit or cfg.system.optim_paged:
            import bitsandbytes as bnb
            if cfg.system.optim_8bit and cfg.system.optim_paged:
                cls = bnb.optim.PagedAdamW8bit
            elif cfg.system.optim_8bit:
                cls = bnb.optim.AdamW8bit
            else:
                cls = bnb.optim.PagedAdamW32bit
            self.optimizer = cls(groups, lr=cfg.task.max_learning_rate, betas=tuple(cfg.task.betas))
        else:
            self.optimizer = torch.optim.AdamW(
                groups, lr=cfg.task.max_learning_rate, betas=tuple(cfg.task.betas)
            )

        self.epoch_steps = len(self.train_loader)
        self.total_steps = self.epoch_steps * cfg.task.epochs
        warmup_steps = cfg.task.get("warmup_steps")
        if warmup_steps is None:
            warmup_steps = int(self.total_steps * cfg.task.get("warmup_ratio", 0.0))
        # WD schedule: warm up, then decay for the entire remaining training
        # run. There is deliberately no stable phase or separate decay span.
        self.scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
            min_lr=cfg.task.min_learning_rate,
        )

        if cfg.task.compile_model:
            self.model = torch.compile(
                self.model, mode=cfg.task.get("compile_mode", "default"), dynamic=True
            )

    @property
    def raw_model(self):
        return self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

    def _save_checkpoint(self, path):
        os.makedirs(path, exist_ok=True)
        self.raw_model.save_pretrained(path, safe_serialization=True)
        config_dir = os.path.join(path, "configs")
        os.makedirs(config_dir, exist_ok=True)
        OmegaConf.save(self.cfg, os.path.join(config_dir, "config.yaml"), resolve=True)

    def _to_device(self, batch):
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    @torch.no_grad()
    def validate(self, epoch, global_step, progress):
        self.model.eval()
        metrics = SegmentationMetrics(self.cfg.task.threshold)
        total_loss = steps = 0
        task = progress.add_task(
            f"[cyan]Validating Epoch {epoch + 1}/{self.cfg.task.epochs}",
            total=len(self.val_loader), postfix="",
        )
        for batch in self.val_loader:
            batch = self._to_device(batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                output = self.model(**batch, **self.loss_kwargs)
            total_loss += output.loss.item()
            steps += 1
            metrics.update(output.logits, batch["labels"], batch["attention_mask"])
            progress.update(task, advance=1, postfix=f"[red]loss: {output.loss.item():.4f}")
        progress.remove_task(task)
        result = {"loss": total_loss / max(steps, 1), **metrics.compute()}
        if self.log_cfg.log_with_wandb:
            wandb.log({f"val/{key}": value for key, value in result.items()}, step=global_step)
        self.model.train()
        return result

    def train(self):
        global_step = 0
        progress = _progress(self.cfg)
        with progress:
            epochs_task = progress.add_task("[yellow]Epochs", total=self.cfg.task.epochs, postfix="")
            for epoch in range(self.cfg.task.epochs):
                train_task = progress.add_task(
                    f"[green]Training Epoch {epoch + 1}/{self.cfg.task.epochs}",
                    total=self.epoch_steps, postfix="[red]loss: N/A",
                )
                self.model.train()
                for batch in self.train_loader:
                    batch = self._to_device(batch)
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                        output = self.model(**batch, **self.loss_kwargs)
                    output.loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), self.cfg.task.gradient_clip_val
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    global_step += 1
                    progress.update(train_task, advance=1, postfix=f"[red]loss: {output.loss.item():.4f}")
                    if self.log_cfg.log_with_wandb:
                        wandb.log({
                            "train/loss": output.loss.item(),
                            "train/adamw_lr": self.optimizer.param_groups[0]["lr"],
                            "train/grad_norm": grad_norm.item(),
                        }, step=global_step)
                    if global_step % self.log_cfg.val_interval == 0 or global_step == self.total_steps:
                        result = self.validate(epoch, global_step, progress)
                        progress.update(epochs_task, postfix=f"[red]val_loss={result['loss']:.4f}")
                progress.remove_task(train_task)
                progress.update(epochs_task, advance=1)
                if (epoch + 1) % self.log_cfg.save_interval == 0:
                    self._save_checkpoint(os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1}"))
        self._save_checkpoint(os.path.join(self.checkpoint_dir, "final_model"))
        if self.log_cfg.log_with_wandb:
            wandb.finish()
