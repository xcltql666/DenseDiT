import lightning as L
from PIL import Image
import numpy as np
import torch
import os
from diffusers.utils import load_image
import random

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        super().__init__()  # 必须调用父类初始化
        
        self.run_name = run_name
        self.training_config = training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0

    def state_dict(self):
        """返回回调的状态字典"""
        return {
            "total_steps": self.total_steps,
        }
    
    def load_state_dict(self, state_dict):
        """从状态字典加载回调状态"""
        self.total_steps = state_dict.get("total_steps", 0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 计算梯度大小
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # 记录到 WandB
        if self.use_wandb and trainer.is_global_zero:
            report_dict = {
                "batch": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            if hasattr(pl_module, "last_t"):
                report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        # 打印训练进度
        if self.total_steps % self.print_every_n_steps == 0 and trainer.is_global_zero:
            loss_value = pl_module.log_loss if hasattr(pl_module, "log_loss") else outputs["loss"].item()
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, "
                f"Loss: {loss_value:.4f}, Gradient size: {gradient_size:.4f}, "
                f"Max gradient size: {max_gradient_size:.4f}"
            )

        # 保存检查点
        if self.total_steps % self.save_interval == 0 and trainer.is_global_zero:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving checkpoint"
            )
            checkpoint_path = f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            os.makedirs(checkpoint_path, exist_ok=True)
            trainer.save_checkpoint(f"{checkpoint_path}/checkpoint.ckpt")

        # 生成样本
        if self.total_steps % self.sample_interval == 0 and trainer.is_global_zero:
            try:
                print(
                    f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating samples"
                )
                self.generate_samples(
                    trainer,
                    pl_module,
                    f"{self.save_path}/{self.run_name}/output",
                    f"sample_{self.total_steps}",
                )
            except Exception as e:
                print(f"Error generating sample: {e}")
                import traceback
                traceback.print_exc()

    @torch.no_grad()
    def generate_samples(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
    ):
        """生成样本图像"""
        # 确保只在主进程中执行
        if not trainer.is_global_zero:
            return
            
        os.makedirs(save_path, exist_ok=True)

        # 使用模型的设备
        device = next(pl_module.parameters()).device
        generator = torch.Generator(device=device).manual_seed(42)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 加载描述文件
        descriptions = []
        description_file = self.training_config.get(
            "description_file",
            "/home/users/astar/ares/qianhang/scratch/chengyou/sx/sx_data/rectify/image_descriptions.txt"
        )
        
        try:
            with open(description_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        columns = line.split(" ", 1)
                        if len(columns) >= 2:
                            first_col = columns[0]
                            second_col = columns[1]
                            descriptions.append((first_col, second_col))
        except Exception as e:
            print(f"Error loading descriptions: {e}")
            return
        
        # 随机选择样本
        random.seed(42)
        num_samples = min(5, len(descriptions))
        selected_descriptions = random.sample(descriptions, num_samples)
        print(f"从 {len(descriptions)} 个例子中随机选择了 {len(selected_descriptions)} 个进行采样")
        
        # 获取图片路径配置
        pairs_dir = self.training_config.get(
            "image_dir",
            "/home/users/astar/ares/qianhang/scratch/chengyou/sx/sx_data/rectify/pairs"
        )
        condition_dir = self.training_config.get(
            "condition_dir",
            "/home/users/astar/ares/qianhang/scratch/chengyou/sx/sx_data/rectify/pairs_pf"
        )
        
        # 生成样本
        for i, (file_name_base, description) in enumerate(selected_descriptions):
            try:
                # 构建图片路径
                context_path = f"{pairs_dir}/{file_name_base}.jpg"
                
                # 处理文件名
                tail = file_name_base.split('_')[-1]
                if tail == 'right':
                    condition_file = file_name_base.replace('_right', '_left')
                else:
                    condition_file = file_name_base.replace('_left', '_right')
                
                condition_path = f"{condition_dir}/{condition_file}_pf.jpg"
                
                # 检查文件是否存在
                if not os.path.exists(condition_path) or not os.path.exists(context_path):
                    print(f"跳过 {file_name_base}: 图片文件不存在")
                    continue

                # 加载输入图片
                condition_img = load_image(condition_path)
                context_img = load_image(context_path)

                # 创建 Condition 对象
                condition = Condition(
                    condition=condition_img,
                    context=context_img,
                )
                
                # 生成图像
                res = generate(
                    pl_module.flux_pipe,
                    prompt=description,
                    conditions=[condition],
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    generator=generator,
                    model_config=pl_module.model_config,
                    default_lora=True,
                )

                # 保存生成的图像
                out_path = os.path.join(save_path, f"{file_name_base}_{self.total_steps}.png")
                res.images[0].save(out_path)
                print(f"保存样本 {i+1}/{len(selected_descriptions)}: {out_path}")
                
                # 清理内存
                del res
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"处理 {file_name_base} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"样本生成完成: {len(selected_descriptions)} 张图片")