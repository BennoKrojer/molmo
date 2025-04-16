import logging
import sys
from pathlib import Path

import torch
import wandb

from olmo.config import CheckpointType, TrainConfig
from olmo.data import build_train_dataloader
from olmo.eval import build_loss_evaluators, build_inf_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import Molmo
from olmo.optim import BoltOnWarmupScheduler, build_optimizer, build_scheduler, build_multimodal_scheduler
from olmo.torch_util import (
    get_default_device,
    seed_all,
    freeze_parameters_by_name,
)
from olmo.train_single_gpu_for_debugging import Trainer
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("train")

def main(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size

    log.info("Configuration:")
    log.info(cfg)

    if cfg.wandb is not None:
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=str(wandb_dir),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=cfg.asdict(exclude=["wandb"]),
        )

    seed_all(cfg.seed)

    train_loader = build_train_dataloader(cfg, device)
    evaluators = build_loss_evaluators(cfg, device) if cfg.eval_interval > 0 or cfg.eval_on_load else None
    inf_evaluators = build_inf_evaluators(cfg, device)

    olmo_model = Molmo(cfg.model)

    # Optional: freeze parts
    if cfg.model.vision_backbone and not cfg.ft_connector:
        freeze_parameters_by_name(olmo_model, Molmo.get_connector_parameters(), warn=False)
    if cfg.model.vision_backbone and not cfg.ft_vit:
        freeze_parameters_by_name(olmo_model, Molmo.get_vit_parameters(), warn=False)
    if not cfg.ft_llm:
        freeze_parameters_by_name(olmo_model, Molmo.get_llm_parameters(), warn=False)

    olmo_model.set_activation_checkpointing(cfg.activation_checkpointing)

    if cfg.initial_model_checkpoint:
        state_dict = torch.load(Path(cfg.initial_model_checkpoint) / "model.pt", map_location="cpu")
        olmo_model.load_state_dict(state_dict)
        del state_dict

    olmo_model.to(device)

    log.info(f"Total parameters: {olmo_model.num_params():,d}")

    optim = build_optimizer(cfg, olmo_model)
    scheduler = build_multimodal_scheduler(cfg) if cfg.model.vision_backbone else build_scheduler(cfg)

    with Trainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=olmo_model,
        fsdp_model=olmo_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=evaluators,
        inference_evaluators=inf_evaluators,
    ) as trainer:
        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1,
            init_method="file:///tmp/tmp_pg"
        )
    prepare_cli_environment()
    add_cached_path_clients()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
