import torch
from olmo.config import TrainConfig
from olmo.checkpoint import build_sharded_checkpointer
from olmo.torch_util import get_local_rank

# --- Config
step_dir = "/mnt/research/scratch/bkroje/molmo_data/molmo_data/checkpoints/train_mlp-only_pixmo_cap_overlap-and-resize-c2/step6000"
cfg_path = step_dir + "/config.yaml"
cfg = TrainConfig.load(cfg_path)
device = torch.device("cuda", get_local_rank())

# --- Patch torch.load to allow full pickle (safe in your case)
torch_load = torch.load  # keep reference
def patched_load(*args, **kwargs):
    return torch_load(*args, weights_only=False, **kwargs)
torch.load = patched_load

# --- Unshard checkpoint (model only)
checkpointer = build_sharded_checkpointer(cfg)
model_state, _, _ = checkpointer.unshard_checkpoint(
    step_dir,
    device=device,
    load_optimizer_state=False,
    load_trainer_state=False,
)

# --- Save as regular .pt file
torch.save(model_state, step_dir + "/model.pt")
