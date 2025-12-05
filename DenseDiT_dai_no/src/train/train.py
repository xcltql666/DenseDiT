import lightning as L
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.callbacks import ModelSummary
from torch.utils.data import DataLoader
import torch
import yaml
import os
import time
import json


from .callbacks import TrainingCallback
from .data import DenseDiTDataset
from .model import DenseDiTModel
from .callbacks import TrainingCallback
from ..utils.memory_tracker import MemoryProfiler


def get_rank():
    """获取当前进程的 rank（Lightning 会自动设置环境变量）"""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size():
    """获取总进程数（Lightning 会自动设置环境变量）"""
    return int(os.environ.get("WORLD_SIZE", 1))


def load_config():
    """加载统一配置文件"""
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_deepspeed_config(config, world_size):
    """
    根据简化配置构建完整的 DeepSpeed 配置
    
    Args:
        config: 统一配置字典
        world_size: GPU 数量
    """
    model_config = config["model"]
    train_config = config["train"]
    ds_config = config["deepspeed"]
    
    # 获取 ZeRO stage 和 offload 配置
    zero_stage = ds_config.get("zero_stage", 3)
    offload = ds_config.get("offload", "none")
    
    # 基础 ZeRO 配置
    zero_optimization = {
        "stage": zero_stage,
        "overlap_comm": ds_config.get("overlap_comm", True),
        "contiguous_gradients": ds_config.get("contiguous_gradients", True),
    }
    
    # 根据 offload 设置配置 offload 策略
    if offload == "none":
        zero_optimization["offload_optimizer"] = {"device": "none"}
        zero_optimization["offload_param"] = {"device": "none"}
    elif offload == "optimizer":
        zero_optimization["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
        zero_optimization["offload_param"] = {"device": "none"}
    elif offload == "all":
        zero_optimization["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
        zero_optimization["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # Stage 3 特定配置
    if zero_stage == 3:
        zero_optimization.update({
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        })
    # Stage 2 特定配置
    elif zero_stage == 2:
        zero_optimization.update({
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
        })
    
    # 混合精度配置
    mixed_precision = model_config.get("mixed_precision", "bfloat16")
    
    # 构建完整配置
    deepspeed_config = {
        "zero_optimization": zero_optimization,
        "fp16": {
            "enabled": mixed_precision == "float16",
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": mixed_precision == "bfloat16"
        },
        "gradient_clipping": train_config.get("gradient_clip_val", 1.0),
        "train_batch_size": train_config["batch_size"] * world_size * train_config["accumulate_grad_batches"],
        "train_micro_batch_size_per_gpu": train_config["batch_size"],
        "wall_clock_breakdown": ds_config.get("wall_clock_breakdown", False),
        "steps_per_print": ds_config.get("steps_per_print", 10),
        "dump_state": False,
    }
    
    return deepspeed_config


def print_training_info(rank, world_size, run_name, config, ds_config):
    """打印训练信息"""
    model_config = config["model"]
    train_config = config["train"]
    
    print("=" * 80)
    print(f"Configurations:")
    print(json.dumps(config, indent=2))
    print("=" * 80)
    print(f"Training Configuration:")
    print(f"  Rank: {rank}/{world_size}")
    print(f"  Run name: {run_name}")
    print(f"  Model path: {model_config['flux_path']}")
    print(f"  Model dtype: {model_config.get('model_dtype', 'bfloat16')}")
    print(f"  Mixed precision: {model_config.get('mixed_precision', 'bfloat16')}")
    print(f"  Batch size per GPU: {train_config['batch_size']}")
    print(f"  Gradient accumulation: {train_config['accumulate_grad_batches']}")
    print(f"  Effective batch size: {ds_config['train_batch_size']}")
    print(f"  DeepSpeed ZeRO Stage: {ds_config['zero_optimization']['stage']}")
    print(f"  Offload strategy: {config['deepspeed'].get('offload', 'none')}")
    print(f"  Training precision: {'BF16 mixed' if ds_config['bf16']['enabled'] else 'FP16 mixed' if ds_config['fp16']['enabled'] else 'FP32'}")
    print(f"  Gradient checkpointing: {train_config.get('gradient_checkpointing', False)}")
    print("=" * 80)


def init_wandb(wandb_config, run_name, config):
    """初始化 Weights & Biases"""
    import wandb
    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config=config,
        )
        return True
    except Exception as e:
        print(f"Failed to initialize WanDB: {e}")
        return False


def main():
    # 获取分布式环境信息
    rank = get_rank()
    world_size = get_world_size()
    is_main_process = (rank == 0)
    
    # 加载统一配置
    config = load_config()
    model_config = config["model"]
    train_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config['train'].get("allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True

    # 构建 DeepSpeed 配置
    deepspeed_config = build_deepspeed_config(config, world_size)
    
    # 打印训练信息
    if is_main_process:
        print_training_info(rank, world_size, run_name, config, deepspeed_config)

    # 初始化 WandB
    wandb_config = train_config.get("wandb")
    if wandb_config and wandb_config.get("enabled") and is_main_process:
        init_wandb(wandb_config, run_name, config)
    
    # 初始化 DeepSpeed 策略
    strategy = DeepSpeedStrategy(
        config=deepspeed_config,
        logging_batch_size_per_gpu=train_config["batch_size"],
    )
    
    # 初始化 Memory Profiler（仅主进程）
    memory_profiler = None
    if is_main_process:
        mem_logfile = f'memory_{run_name}.log'
        memory_profiler = MemoryProfiler(
            enable_tensor_accumulation=True, 
            log_file=mem_logfile
        )
        memory_profiler.snapshot("start")

    # ===== 初始化数据集 =====
    # 注意：Lightning 会自动处理分布式采样，不需要手动创建 DistributedSampler
    def load_descriptions(description_file):
        descriptions = {}
        with open(description_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or " " not in line:
                    continue
                file_name, description = line.split(" ", 1)
                descriptions[file_name] = description
        return descriptions
    
    descriptions = load_descriptions(train_config["description_file"])
    
    dataset = DenseDiTDataset(
        image_dir=train_config["image_dir"],
        condition_dir=train_config["condition_dir"],
        context_file=train_config["context_file"],
        descriptions=descriptions,
        resize=train_config.get("image_size", (512, 512))
    )

    if is_main_process:
        print(f"Dataset length: {len(dataset)}")
        if memory_profiler:
            memory_profiler.snapshot("after_dataset_load")
    
    # Lightning 会自动处理分布式采样，只需创建普通的 DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,  # Lightning 会自动转换为分布式采样
        num_workers=train_config["dataloader_workers"],
        pin_memory=True,
        drop_last=True,
    )

    if is_main_process and memory_profiler:
        memory_profiler.snapshot("after_dataloader")

    # ===== 初始化模型 =====
    trainable_model = DenseDiTModel(
        flux_pipe_id=model_config["flux_path"],
        train_dtype=model_config.get("model_dtype", "float32"),
        inference_dtype=model_config.get("mixed_precision", "bfloat16"),
        optimizer_config=train_config["optimizer"],
        model_config=model_config,
        gradient_checkpointing=train_config.get("gradient_checkpointing", False)
    )
    trainable_model.train()

    print("Model parameter on ", next(trainable_model.parameters()).device)
    if is_main_process and memory_profiler:
        memory_profiler.register_model(trainable_model, "DenseDiTModel")
        memory_profiler.snapshot("after_model_load")

    # ===== 初始化 Callbacks =====
    training_callbacks = [ModelSummary(max_depth=0)]  # 禁用默认的模型 summary
    if is_main_process:
        # 添加训练回调
        training_callbacks.append(
            TrainingCallback(run_name, training_config=train_config)
        )

    # ===== 初始化 Trainer =====
    # 设置混合精度
    mixed_precision = model_config.get("mixed_precision", "bfloat16")
    if mixed_precision == "bfloat16":
        precision = "bf16-mixed"
    elif mixed_precision == "float16":
        precision = "16-mixed"
    else:
        precision = "32"
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=world_size,
        accumulate_grad_batches=train_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=True,
        enable_progress_bar=is_main_process,
        logger=True,
        strategy=strategy,
        max_steps=train_config.get("max_steps", -1),
        max_epochs=train_config.get("max_epochs", -1),
        gradient_clip_val=train_config.get("gradient_clip_val", 1.0),
        precision=precision,
        log_every_n_steps=train_config.get("log_every_n_steps", 50),
    )

    if is_main_process and memory_profiler:
        memory_profiler.snapshot("after_trainer_init")
    
    # 附加训练配置到 trainer
    setattr(trainer, "training_config", train_config)

    # ===== 保存配置 =====
    # save_path = train_config.get("save_path", "./output")
    # if is_main_process:
    #     run_dir = f"{save_path}/{run_name}"
    #     os.makedirs(run_dir, exist_ok=True)
        
    #     # 保存配置
    #     with open(f"{run_dir}/config.yaml", "w") as f:
    #         yaml.dump(config, f)
        
    #     # 保存生成的 DeepSpeed 配置（用于调试）
    #     with open(f"{run_dir}/deepspeed_config.json", "w") as f:
    #         json.dump(deepspeed_config, f, indent=2)
        
    #     print(f"Configurations saved to: {run_dir}")

    if is_main_process and memory_profiler:
        memory_profiler.snapshot("before_training")
        memory_profiler.print_full_report()
    
    # ===== 开始训练 =====
    zero_stage = config["deepspeed"].get("zero_stage", 3)
    offload = config["deepspeed"].get("offload", "none")
    print(f"Starting training with DeepSpeed ZeRO Stage {zero_stage} (offload: {offload})...")
    trainer.fit(trainable_model, train_loader)  # 取消注释以开始训练


if __name__ == "__main__":
    main()