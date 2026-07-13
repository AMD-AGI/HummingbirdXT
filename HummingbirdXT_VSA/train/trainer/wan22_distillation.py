# download from https://github.com/guandeh17/Self-Forcing
import gc
import json
import logging
import re

from utils.dataset import ODERegressionCSVDataset, cycle, OffsetDistributedSampler
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.benchmark_warmup_steps = int(
            getattr(config, "benchmark_warmup_steps", 0) or 0)
        self.benchmark_steps = int(getattr(config, "benchmark_steps", 0) or 0)
        self.benchmark_output = getattr(config, "benchmark_output", "")
        self.benchmark_enabled = self.benchmark_steps > 0
        self.benchmark_records = []

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            run_name = config.config_name
            if getattr(config, "logdir", ""):
                normalized_logdir = os.path.normpath(str(config.logdir))
                logdir_name = os.path.basename(normalized_logdir)
                if logdir_name:
                    run_name = logdir_name
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=run_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = DMD(config, device=self.device)

        # Resume Training from Latest Checkpoint
        pretrained_ckpt_path, self.step, pretrained_state_dict = (
            (None, 0, None) if self.benchmark_enabled else self.load(self.output_path))
        if pretrained_ckpt_path is not None:
            if self.is_main_process:
                print(f"Loading checkpoint from {pretrained_ckpt_path} at step {self.step}")
            self.model.generator.load_state_dict(
                pretrained_state_dict["generator"], strict=True)
            self.model.fake_score.load_state_dict(
                pretrained_state_dict["critic"], strict=True)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = ODERegressionCSVDataset(
            config.data_path, 
            max_pair=int(1e8), 
            num_frames=config.num_frames,
            h=config.h,
            w=config.w,
        )
            
        sampler = OffsetDistributedSampler(
            dataset,
            initial_step=self.step,
            gpu_num=self.world_size,
            shuffle=False,
            drop_last=True,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        self.ema_weight = config.get("ema_weight", -1.0)
        self.ema_start_step = config.get("ema_start_step", 0)
        self.generator_ema = None
        if (self.ema_weight > 0.0) and (self.step >= self.ema_start_step):
            print(f"Setting up EMA with weight {self.ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)
            
            # Load EMA state dict if available in checkpoint
            if pretrained_state_dict is not None:
                if "generator_ema" in pretrained_state_dict:
                    print("Loading generator_ema from checkpoint")
                    self.generator_ema.load_state_dict(
                        pretrained_state_dict["generator_ema"])
                else:
                    print("No generator_ema found in checkpoint, starting fresh EMA")

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        # if self.step < config.ema_start_step:
        #     self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def load(self, out_path):
        # 1. 找到最新的checkpoint文件夹（按步数排序）
        if not os.path.exists(out_path):
            return None, 0, None
        ckpt_folders = [f for f in os.listdir(out_path) if f.startswith("checkpoint_model_")]
        if not ckpt_folders:
            return None, 0, None
        # 提取步数
        def extract_step(folder_name):
            match = re.search(r"checkpoint_model_(\d+)", folder_name)
            return int(match.group(1)) if match else -1
        ckpt_folders.sort(key=extract_step, reverse=True)

        # 2. 优先尝试最新checkpoint；若损坏则自动回退
        for folder in ckpt_folders:
            model_path = os.path.join(out_path, folder, "model.pt")
            if not os.path.exists(model_path):
                continue
            step = extract_step(folder)
            try:
                checkpoint_state_dict = torch.load(model_path, map_location="cpu")
            except Exception as e:
                if self.is_main_process:
                    print(f"Skip corrupted checkpoint {model_path}: {e}")
                continue
            return model_path, step, checkpoint_state_dict

        if self.is_main_process:
            print(f"No valid checkpoint found under {out_path}, starting from scratch.")
        return None, 0, None

    def _atomic_torch_save(self, state_dict, save_path: str) -> None:
        tmp_path = f"{save_path}.tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, save_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _measure_cuda_ms(self, fn):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = fn()
        end_event.record()
        torch.cuda.synchronize()
        return result, start_event.elapsed_time(end_event)

    def _max_time_tensor(self, local_ms):
        time_tensor = torch.tensor(
            float(local_ms), device=self.device, dtype=torch.float32)
        if self.benchmark_enabled and dist.is_initialized():
            dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
        return time_tensor

    def _tensor_to_float(self, value, default=0.0):
        if value is None:
            return float(default)
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().cpu().item())
        return float(value)

    def _unwrap_module(self, module):
        current = module
        visited = set()
        while isinstance(current, torch.nn.Module):
            current_id = id(current)
            if current_id in visited:
                break
            visited.add(current_id)

            next_module = None
            for attr_name in ("module", "_fsdp_wrapped_module"):
                candidate = getattr(current, attr_name, None)
                if isinstance(candidate, torch.nn.Module):
                    next_module = candidate
                    break

            if next_module is None:
                break
            current = next_module

        return current

    def _reset_trainable_forward_timing(self, module):
        wrapper = self._unwrap_module(module)
        if hasattr(wrapper, "reset_trainable_forward_timing"):
            wrapper.reset_trainable_forward_timing()

    def _consume_trainable_forward_ms(self, module):
        wrapper = self._unwrap_module(module)
        if hasattr(wrapper, "consume_trainable_forward_ms"):
            return wrapper.consume_trainable_forward_ms()
        return 0.0

    def _summarize_scalar_list(self, values):
        if not values:
            return None
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    def _build_benchmark_summary(self):
        metric_names = [
            "forward_ms",
            "loss_forward_ms",
            "backward_ms",
            "fwdbwd_wall_ms",
            "generator_forward_ms",
            "generator_loss_forward_ms",
            "generator_backward_ms",
            "critic_forward_ms",
            "critic_loss_forward_ms",
            "critic_backward_ms",
        ]
        metrics = {}
        for metric_name in metric_names:
            metric_values = [record[metric_name] for record in self.benchmark_records]
            metrics[metric_name] = self._summarize_scalar_list(metric_values)

        generator_step_records = [
            record for record in self.benchmark_records if record["train_generator"]
        ]
        generator_step_wall_ms = [
            record["fwdbwd_wall_ms"] for record in generator_step_records
        ]
        critic_only_step_wall_ms = [
            record["fwdbwd_wall_ms"] for record in self.benchmark_records
            if not record["train_generator"]
        ]

        return {
            "backend": os.environ.get("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN"),
            "world_size": self.world_size,
            "warmup_steps": self.benchmark_warmup_steps,
            "measured_steps": len(self.benchmark_records),
            "dfake_gen_update_ratio": int(self.config.dfake_gen_update_ratio),
            "metrics": metrics,
            "generator_step_wall_ms": self._summarize_scalar_list(
                generator_step_wall_ms),
            "critic_only_step_wall_ms": self._summarize_scalar_list(
                critic_only_step_wall_ms),
            "records": self.benchmark_records,
        }

    def _write_benchmark_summary(self, summary):
        if not self.benchmark_output:
            return
        output_dir = os.path.dirname(self.benchmark_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.benchmark_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _print_benchmark_summary(self, summary):
        if not self.is_main_process:
            return
        metrics = summary["metrics"]
        print("Benchmark summary:")
        print(f"  backend: {summary['backend']}")
        print(f"  measured steps: {summary['measured_steps']}")
        print(
            "  avg trainable forward/backward/wall (ms): "
            f"{metrics['forward_ms']['mean']:.2f} / "
            f"{metrics['backward_ms']['mean']:.2f} / "
            f"{metrics['fwdbwd_wall_ms']['mean']:.2f}"
        )
        if metrics["loss_forward_ms"] is not None:
            print(
                "  avg whole-loss forward (ms): "
                f"{metrics['loss_forward_ms']['mean']:.2f}"
            )
        if summary["generator_step_wall_ms"] is not None:
            print(
                "  avg generator+critic step wall (ms): "
                f"{summary['generator_step_wall_ms']['mean']:.2f}"
            )
        if summary["critic_only_step_wall_ms"] is not None:
            print(
                "  avg critic-only step wall (ms): "
                f"{summary['critic_only_step_wall_ms']['mean']:.2f}"
            )
        if self.benchmark_output:
            print(f"  benchmark json: {self.benchmark_output}")
    
    def save(self):
        if dist.is_initialized():
            dist.barrier()
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        if dist.is_initialized():
            dist.barrier()
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)
        if dist.is_initialized():
            dist.barrier()

        if (self.ema_weight > 0.0) and (self.ema_start_step < self.step):
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            save_dir = os.path.join(
                self.output_path, f"checkpoint_model_{self.step:06d}")
            save_path = os.path.join(save_dir, "model.pt")
            os.makedirs(save_dir, exist_ok=True)
            self._atomic_torch_save(state_dict, save_path)
            print("Model saved to", save_path)
        if dist.is_initialized():
            dist.barrier()

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        video_tensor = batch["video"].to(device=self.device, dtype=self.dtype)
        first_frame = video_tensor[:, :, :1, :, :]
        wan22_image_latent = self.model.vae.encode_to_latent(first_frame) # torch.Size([1, 1, 48, 44, 80])
        clean_latent = None
        image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            if self.config.i2v:
                img = batch["img"].to(self.device).squeeze(0)
                clip_fea = self.model.image_encoder(img)
                y = self.model.vae.run_vae_encoder(img)
            else:
                clip_fea = None
                y = None

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            #print(image_or_video_shape)
            if self.benchmark_enabled:
                self._reset_trainable_forward_timing(self.model.generator)
            if self.benchmark_enabled:
                (generator_loss, generator_log_dict), generator_loss_forward_ms = self._measure_cuda_ms(
                    lambda: self.model.generator_loss(
                        image_or_video_shape=image_or_video_shape,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        clean_latent=clean_latent,
                        initial_latent=image_latent if self.config.i2v else None,
                        clip_fea=clip_fea,
                        y=y,
                        wan22_image_latent=wan22_image_latent,
                    )
                )
                generator_forward_ms = self._consume_trainable_forward_ms(
                    self.model.generator)
            else:
                generator_loss, generator_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=image_latent if self.config.i2v else None,
                    clip_fea=clip_fea,
                    y=y,
                    wan22_image_latent=wan22_image_latent,
                )
                generator_forward_ms = 0.0
                generator_loss_forward_ms = 0.0

            torch.cuda.empty_cache()

            if self.benchmark_enabled:
                _, generator_backward_ms = self._measure_cuda_ms(
                    lambda: generator_loss.backward())
            else:
                generator_loss.backward()
                generator_backward_ms = 0.0
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({
                "generator_loss": generator_loss,
                "generator_grad_norm": generator_grad_norm,
                "generator_forward_ms": self._max_time_tensor(
                    generator_forward_ms),
                "generator_loss_forward_ms": self._max_time_tensor(
                    generator_loss_forward_ms),
                "generator_backward_ms": self._max_time_tensor(
                    generator_backward_ms),
            })

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        if self.benchmark_enabled:
            self._reset_trainable_forward_timing(self.model.fake_score)
        if self.benchmark_enabled:
            (critic_loss, critic_log_dict), critic_loss_forward_ms = self._measure_cuda_ms(
                lambda: self.model.critic_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=image_latent if self.config.i2v else None,
                    clip_fea=clip_fea,
                    y=y,
                    wan22_image_latent=wan22_image_latent,
                )
            )
            critic_forward_ms = self._consume_trainable_forward_ms(
                self.model.fake_score)
        else:
            critic_loss, critic_log_dict = self.model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                clip_fea=clip_fea,
                y=y,
                wan22_image_latent=wan22_image_latent,
            )
            critic_forward_ms = 0.0
            critic_loss_forward_ms = 0.0

        if self.benchmark_enabled:
            _, critic_backward_ms = self._measure_cuda_ms(
                lambda: critic_loss.backward())
        else:
            critic_loss.backward()
            critic_backward_ms = 0.0
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({
            "critic_loss": critic_loss,
            "critic_grad_norm": critic_grad_norm,
            "critic_forward_ms": self._max_time_tensor(critic_forward_ms),
            "critic_loss_forward_ms": self._max_time_tensor(
                critic_loss_forward_ms),
            "critic_backward_ms": self._max_time_tensor(critic_backward_ms),
        })

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step

        while True:
            if self.is_main_process:
                print(f"training step {self.step} ...")
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            torch.cuda.synchronize()
            iter_fwdbwd_start = time.time()
            generator_log_dict = {}

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                if not self.config.debug:
                    self.generator_optimizer.step()
                    if self.generator_ema is not None:
                        self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            if not self.config.debug:
                self.critic_optimizer.step()

            torch.cuda.synchronize()
            iter_fwdbwd_time = time.time() - iter_fwdbwd_start
            iter_fwdbwd_wall_ms = iter_fwdbwd_time * 1000.0

            # Increment the step since we finished gradient update
            self.step += 1

            generator_forward_ms = self._tensor_to_float(
                generator_log_dict.get("generator_forward_ms"))
            generator_loss_forward_ms = self._tensor_to_float(
                generator_log_dict.get("generator_loss_forward_ms"))
            generator_backward_ms = self._tensor_to_float(
                generator_log_dict.get("generator_backward_ms"))
            critic_forward_ms = self._tensor_to_float(
                critic_log_dict.get("critic_forward_ms"))
            critic_loss_forward_ms = self._tensor_to_float(
                critic_log_dict.get("critic_loss_forward_ms"))
            critic_backward_ms = self._tensor_to_float(
                critic_log_dict.get("critic_backward_ms"))
            iter_forward_ms = generator_forward_ms + critic_forward_ms
            iter_loss_forward_ms = (
                generator_loss_forward_ms + critic_loss_forward_ms)
            iter_backward_ms = generator_backward_ms + critic_backward_ms

            # Create EMA params (if not already created)
            if (self.step >= self.ema_start_step) and \
                    (self.generator_ema is None) and (self.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                print(f"  step {self.step} fwd+bwd time: {iter_fwdbwd_time:.3f}s")
                wandb_loss_dict = {"fwdbwd_time": iter_fwdbwd_time}
                if self.benchmark_enabled:
                    print(
                        "  step timings (ms): "
                        f"trainable_forward={iter_forward_ms:.2f}, "
                        f"loss_forward={iter_loss_forward_ms:.2f}, "
                        f"backward={iter_backward_ms:.2f}, "
                        f"wall={iter_fwdbwd_wall_ms:.2f}"
                    )
                    wandb_loss_dict.update(
                        {
                            "forward_ms": iter_forward_ms,
                            "loss_forward_ms": iter_loss_forward_ms,
                            "backward_ms": iter_backward_ms,
                            "fwdbwd_wall_ms": iter_fwdbwd_wall_ms,
                        }
                    )
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            completed_steps = self.step - start_step
            if self.benchmark_enabled and completed_steps > self.benchmark_warmup_steps:
                self.benchmark_records.append(
                    {
                        "step": int(self.step),
                        "train_generator": bool(TRAIN_GENERATOR),
                        "forward_ms": iter_forward_ms,
                        "loss_forward_ms": iter_loss_forward_ms,
                        "backward_ms": iter_backward_ms,
                        "fwdbwd_wall_ms": iter_fwdbwd_wall_ms,
                        "generator_forward_ms": generator_forward_ms,
                        "generator_loss_forward_ms": generator_loss_forward_ms,
                        "generator_backward_ms": generator_backward_ms,
                        "critic_forward_ms": critic_forward_ms,
                        "critic_loss_forward_ms": critic_loss_forward_ms,
                        "critic_backward_ms": critic_backward_ms,
                    }
                )
                if len(self.benchmark_records) >= self.benchmark_steps:
                    dist.barrier()
                    summary = self._build_benchmark_summary()
                    self._write_benchmark_summary(summary)
                    self._print_benchmark_summary(summary)
                    return
