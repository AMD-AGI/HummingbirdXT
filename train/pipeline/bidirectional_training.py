# Modified from https://github.com/guandeh17/Self-Forcing/tree/main and https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo
# Modifications Copyright(C)[2026] Advanced Micro Devices, Inc. All rights reserved.
from typing import List
import torch

from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
import torch.distributed as dist
from utils.dataset import masks_like

class BidirectionalTrainingPipeline(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
    ):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

    def generate_and_sync_list(self, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(1,),
                device=device
            )
        else:
            indices = torch.empty(1, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(self, noise: torch.Tensor, clip_fea, y, wan22_image_latent, **conditional_dict) -> torch.Tensor:

        # initial point
        noisy_image_or_video = noise
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(num_denoising_steps, device=noise.device)

        # use the last n-1 timesteps to simulate the generator's input
        for index, current_timestep in enumerate(self.denoising_step_list):
            exit_flag = (index == exit_flags[0])
            timestep = torch.ones(
                noise.shape[:2],
                device=noise.device,
                dtype=torch.int64) * current_timestep

            if "2.2" in self.generator.model_name:
                mask1, mask2 = masks_like(noisy_image_or_video, zero=True)
                mask2 = torch.stack(mask2, dim=0) # torch.Size([1, 31, 48, 44, 80])
                noisy_image_or_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_image_or_video
                noisy_image_or_video = noisy_image_or_video.to(noise.device, dtype=noise.dtype)

                wan22_input_timestep = torch.tensor([timestep[0][0].item()], device=noise.device, dtype=noise.dtype)
                temp_ts = (mask2[:, :, 0, ::2, ::2] * wan22_input_timestep)
                temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
                temp_ts = torch.cat([temp_ts, temp_ts.new_ones(self.generator.seq_len - temp_ts.size(1)) * wan22_input_timestep], dim=1)
                wan22_input_timestep = temp_ts.to(noise.device, dtype=torch.long)
            else:
                mask1, mask2 = None, None
                wan22_input_timestep = None


            if not exit_flag:
                with torch.no_grad():
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_image_or_video,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        clip_fea=clip_fea,
                        y=y,
                        wan22_input_timestep=wan22_input_timestep,
                        mask2=mask2,
                        wan22_image_latent=wan22_image_latent,
                    )  # [B, F, C, H, W]

                    next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                        noise.shape[:2], dtype=torch.long, device=noise.device)
                    noisy_image_or_video = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep.flatten(0, 1)
                    ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    clip_fea=clip_fea,
                    y=y,
                    wan22_input_timestep=wan22_input_timestep,
                    mask2=mask2,
                    wan22_image_latent=wan22_image_latent,
                )  # [B, F, C, H, W]
                break

        if exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        return denoised_pred, denoised_timestep_from, denoised_timestep_to
