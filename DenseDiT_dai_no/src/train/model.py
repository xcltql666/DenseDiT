"""
DenseDiT Model for training with Lightning
"""
import lightning as L
from diffusers.pipelines import FluxKontextPipeline
import torch
from peft import LoraConfig, get_peft_model_state_dict

try:
    import prodigyopt
    PRODIGY_AVAILABLE = True
except ImportError:
    PRODIGY_AVAILABLE = False
    print("Warning: prodigyopt not available, Prodigy optimizer will not work")

from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, prepare_text_input

dtype_map = {
    'float64': torch.float64,
    'float': torch.float32,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

class DenseDiTModel(L.LightningModule):
    """
    DenseDiT Model for training with Lightning
    """
    def __init__(
        self,
        flux_pipe_id: str,
        train_dtype: torch.dtype = torch.float32,
        inference_dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = {},
        gradient_checkpointing: bool = False
    ):
        super().__init__()  # 必须首先调用
        if isinstance(train_dtype, str):
            train_dtype = dtype_map.get(train_dtype, torch.float32)
        if isinstance(inference_dtype, str):
            inference_dtype = dtype_map.get(inference_dtype, torch.bfloat16)
        
        # 保存配置
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.train_dtype = train_dtype
        self.inference_dtype = inference_dtype
        
        print(f"Initializing DenseDiTModel...")
        print(f"  dtype: {self.train_dtype}")
        print(f"  gradient_checkpointing: {gradient_checkpointing}")
        
        # 加载 FluxKontext pipeline
        print(f"Loading FluxKontext pipeline from {flux_pipe_id}...")
        self.flux_pipe: FluxKontextPipeline = FluxKontextPipeline.from_pretrained(
            flux_pipe_id,
            torch_dtype=train_dtype
        )
        
        # 获取 transformer
        self.transformer = self.flux_pipe.transformer
        
        # 设置梯度检查点
        if gradient_checkpointing:
            self.transformer.gradient_checkpointing = True
            self.transformer._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
            print("  Gradient checkpointing enabled")
        
        # 保存超参数（用于检查点恢复）
        self.save_hyperparameters(ignore=['flux_pipe_id'])
        
        # 冻结不需要训练的组件
        print("Freezing non-trainable components...")
        self.flux_pipe.text_encoder.requires_grad_(False)
        self.flux_pipe.text_encoder_2.requires_grad_(False)
        self.flux_pipe.vae.requires_grad_(False)
        self.flux_pipe.text_encoder.to(inference_dtype)
        self.flux_pipe.text_encoder_2.to(inference_dtype)
        self.flux_pipe.vae.to(train_dtype)
        self.flux_pipe.transformer.to(train_dtype)
        
        # 设置可训练参数
        self._setup_trainable_params()
        
        print("DenseDiTModel initialized successfully")

    def _setup_trainable_params(self):
        """设置可训练的参数"""
        print("Setting up trainable parameters...")
        
        # 首先冻结整个 transformer
        self.transformer.requires_grad_(False)
        
        # 解冻 transformer_blocks（双流块）
        for block in self.transformer.transformer_blocks:
            for param in block.parameters():
                param.requires_grad = True
        
        # 可选：解冻 single_transformer_blocks（单流块）
        # for block in self.transformer.single_transformer_blocks:
        #     for param in block.parameters():
        #         param.requires_grad = True
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.transformer.parameters())
        
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")

    def configure_optimizers(self):
        """配置优化器 - Lightning 会自动调用"""
        print("Configuring optimizers...")
        
        # 获取所有需要梯度的参数
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found!")
        
        print(f"Number of trainable parameters: {len(trainable_params)}")
        
        opt_config = self.optimizer_config
        opt_type = opt_config.get("type", "AdamW")
        opt_params = opt_config.get("params", {})
        
        # 初始化优化器
        if opt_type == "AdamW":
            optimizer = torch.optim.AdamW(trainable_params, **opt_params)
        elif opt_type == "Prodigy":
            if not PRODIGY_AVAILABLE:
                raise ImportError("prodigyopt is not installed. Install it with: pip install prodigyopt")
            optimizer = prodigyopt.Prodigy(trainable_params, **opt_params)
        elif opt_type == "SGD":
            optimizer = torch.optim.SGD(trainable_params, **opt_params)
        else:
            raise NotImplementedError(f"Optimizer {opt_type} not implemented")
        
        print(f"Optimizer configured: {opt_type}")
        return optimizer

    def training_step(self, batch, batch_idx):
        """训练步骤 - Lightning 会自动调用"""
        step_loss = self.step(batch)
        
        # 记录损失
        self.log('train/loss', step_loss, prog_bar=True, sync_dist=True)
        if hasattr(self, 'last_t'):
            self.log('train/avg_timestep', self.last_t, prog_bar=True, sync_dist=True)
        
        # EMA 平滑损失（用于日志）
        if not hasattr(self, 'log_loss'):
            self.log_loss = step_loss.item()
        else:
            self.log_loss = self.log_loss * 0.95 + step_loss.item() * 0.05
        
        return {"loss": step_loss}

    def step(self, batch):
        """单步训练逻辑"""
        imgs = batch["image"]
        conditions = batch["condition"]
        context = batch["context"]
        prompts = batch["description"]

        # Prepare image input
        x_0, img_ids = encode_images(self.flux_pipe, imgs, device=self.device, dtype=self.train_dtype)

        # Prepare conditions
        condition_latents, condition_ids = encode_images(self.flux_pipe, conditions, device=self.device, dtype=self.train_dtype)
        
        # Prepare context
        context_latents, context_ids = encode_images(self.flux_pipe, context, device=self.device, dtype=self.train_dtype)
        
        # Prepare text input
        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
            self.flux_pipe, prompts, device=self.device, dtype=self.train_dtype
        )

        # Prepare t and x_t
        t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device)) # (B,)
        x_1 = torch.randn_like(x_0, device=self.device, dtype=self.train_dtype) # (B, Seq_len, C)
        t_ = t.view(-1, 1, 1) # (B, 1, 1)
        x_t = ((1 - t_) * x_0 + t_ * x_1)

        # Prepare guidance
        guidance = (
            torch.ones_like(t)
            if self.transformer.config.guidance_embeds
            else None
        )
        
        # Concatenate latents
        latent_ids = torch.cat([img_ids, condition_ids, context_ids], dim=0) # (Seq_len * 3, 3)
        latent_model_input = torch.cat([x_t, condition_latents, context_latents], dim=1)
        
        latent_model_input.requires_grad_(True)
        with torch.autocast(device_type=self.device.type, dtype=self.train_dtype):
            # Forward pass
            transformer_out = self.transformer(
                hidden_states=latent_model_input,
                timestep=t,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )
        
        pred = transformer_out[0]
        pred = pred[:, :x_t.size(1)]  # 裁剪到原始图像大小
        
        # Compute loss
        target = x_1 - x_0
        loss = torch.nn.functional.mse_loss(pred, target, reduction="mean")
        
        # 记录平均时间步
        self.last_t = t.mean().item()
        
        return loss
    
    def on_save_checkpoint(self, checkpoint):
        """保存检查点时的回调"""
        checkpoint['model_config'] = self.model_config
    
    def on_load_checkpoint(self, checkpoint):
        """加载检查点时的回调"""
        if 'model_config' in checkpoint:
            self.model_config = checkpoint['model_config']