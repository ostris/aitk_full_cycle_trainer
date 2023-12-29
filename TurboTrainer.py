import math
import os
from collections import OrderedDict
import random
from typing import Union, List, Optional

import numpy as np
from diffusers import T2IAdapter, AutoencoderTiny
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import v2

from jobs.process.models.vgg19_critic import Critic
from toolkit import train_tools
from toolkit.basic import get_mean_std, value_map
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.image_utils import show_tensors
from PIL import Image
from toolkit.prompt_utils import PromptEmbeds, split_prompt_embeds
from toolkit.sampler import get_sampler
from toolkit.stable_diffusion_model import BlankNetwork
from toolkit.style import get_style_model_and_losses
from toolkit.train_tools import get_torch_dtype, add_all_snr_to_noise_scheduler
import gc
import torch
from jobs.process import BaseSDTrainProcess
from torchvision import transforms


def flush():
    torch.cuda.empty_cache()
    gc.collect()


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])


class SDTurboTrainer(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', None]
        self.do_prior_prediction = False
        self.steps_in_cycle = self.config.get('steps_in_cycle', 4)
        self.unconditional_prompt: PromptEmbeds
        self.taesd: AutoencoderTiny
        if self.train_config.inverted_mask_prior:
            self.do_prior_prediction = True
        self.vgg_19 = None
        self.style_weight = self.config.get('style_weight', 1e1)
        self.content_weight = self.config.get('content_weight', 1e-1)
        self.l2_weight = self.config.get('l2_weight', 1-1)
        self.critic_weight = self.get_conf('critic_weight', 1e-3)
        self.negative_prompt = self.config.get('negative_prompt', '')
        self.style_weight_scalers = []
        self.content_weight_scalers = []
        self.torch_dtype = get_torch_dtype(self.train_config.dtype)
        self.style_losses = []
        self.content_losses = []
        self.sample_mean_std = None
        self.mean_std_cache = []
        self.max_cache_size = 20
        self.vgg19_pool_4 = None
        self.terms_location = self.get_conf('terms_location', None)
        self.use_taesd = self.get_conf('use_taesd', False)
        self.use_vae = self.get_conf('use_vae', True)
        self.taesd = None
        self.use_critic = self.get_conf('use_critic', False)
        self.critic_warmup_steps = self.get_conf('critic_warmup_steps', 500)
        self.dtype = self.train_config.dtype
        self.max_steps = self.train_config.steps
        self.size = self.get_conf('size', 512)
        self.teacher_scheduler_name = self.get_conf('teacher_scheduler', 'ddpm')
        self.teacher_scheduler = get_sampler(self.teacher_scheduler_name)
        self.child_scheduler = get_sampler(self.train_config.noise_scheduler)
        if self.use_critic:
            self.critic = Critic(
                device=self.device,
                dtype=self.dtype,
                process=self,
                **self.get_conf('critic', {})  # pass any other params
            )
        self.guidance_scale = self.get_conf('guidance_scale', 1.0)
        self.num_inference_steps = self.get_conf('num_inference_steps', 20)
        self.num_turbo_steps = self.get_conf('num_turbo_steps', 1)

        self.terms_list: List[str] = []

    def setup_vgg19(self):
        if self.vgg_19 is None:
            self.vgg_19, self.style_losses, self.content_losses, self.vgg19_pool_4 = get_style_model_and_losses(
                single_target=True,
                device=self.device,
                output_layer_name='pool_4',
                dtype=self.torch_dtype
            )
            self.vgg_19.to(self.device, dtype=self.torch_dtype)
            self.vgg_19.requires_grad_(False)

            # we run random noise through first to get layer scalers to normalize the loss per layer
            # bs of 2 because we run pred and target through stacked
            size = self.size
            noise = torch.randn((2, 3, size, size), device=self.device, dtype=self.torch_dtype)
            self.vgg_19(noise)
            for style_loss in self.style_losses:
                # get a scaler  to normalize to 1
                scaler = 1 / torch.mean(style_loss.loss).item()
                self.style_weight_scalers.append(scaler)
            for content_loss in self.content_losses:
                # get a scaler  to normalize to 1
                scaler = 1 / torch.mean(content_loss.loss).item()
                # if is nan, set to 1
                if scaler != scaler:
                    scaler = 1
                    print(f"Warning: content loss scaler is nan, setting to 1")
                self.content_weight_scalers.append(scaler)

            self.print(f"Style weight scalers: {self.style_weight_scalers}")
            self.print(f"Content weight scalers: {self.content_weight_scalers}")
        if self.use_critic:
            self.critic.setup()

    def get_style_loss(self):
        if self.style_weight > 0:
            # scale all losses with loss scalers
            loss = torch.sum(
                torch.stack([loss.loss * scaler for loss, scaler in zip(self.style_losses, self.style_weight_scalers)]))
            return loss
        else:
            return torch.tensor(0.0, device=self.device)

    def get_content_loss(self):
        if self.content_weight > 0:
            # scale all losses with loss scalers
            loss = torch.sum(torch.stack(
                [loss.loss * scaler for loss, scaler in zip(self.content_losses, self.content_weight_scalers)]))
            return loss
        else:
            return torch.tensor(0.0, device=self.device)

    def before_model_load(self):
        pass

    def before_dataset_load(self):
        self.assistant_adapter = None
        # get adapter assistant if one is set
        if self.train_config.adapter_assist_name_or_path is not None:
            adapter_path = self.train_config.adapter_assist_name_or_path

            # dont name this adapter since we are not training it
            self.assistant_adapter = T2IAdapter.from_pretrained(
                adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype), varient="fp16"
            ).to(self.device_torch)
            self.assistant_adapter.eval()
            self.assistant_adapter.requires_grad_(False)
            flush()

    @torch.no_grad()
    def update_sample_mean_std(self):
        sample_folder = os.path.join(self.save_root, 'samples')
        if not os.path.exists(sample_folder):
            return

        num_samples = len(self.sample_config.prompts)
        # find the latest num_samples images in the sample folder, (png or jpg)
        sample_files = [os.path.join(sample_folder, f) for f in os.listdir(sample_folder) if
                        os.path.splitext(f)[1].lower() in ['.png', '.jpg']]
        if sample_files and len(sample_files) > 0:

            sample_files.sort(key=os.path.getmtime)
            sample_files = sample_files[-num_samples:]
            avg_mean_std = None
            for sample_file in sample_files:
                # load as -1 to 1 torch tensor
                img = adapter_transforms(Image.open(sample_file)).to(self.device_torch, dtype=self.torch_dtype)
                mean, std = get_mean_std(img)
                if avg_mean_std is None:
                    avg_mean_std = mean, std
                else:
                    avg_mean_std = avg_mean_std[0] + mean, avg_mean_std[1] + std

            avg_mean_std = avg_mean_std[0] / len(sample_files), avg_mean_std[1] / len(sample_files)

            self.sample_mean_std = avg_mean_std

    # override the sample function to run after it
    def sample(self, step=None, is_first=False):
        super().sample(step, is_first)
        self.update_sample_mean_std()

    def save(self, step=None):
        super().save(step)
        if self.use_critic:
            self.critic.save(step)

    def hook_before_train_loop(self):
        # move vae to device if we did not cache latents
        if not self.is_latents_cached:
            self.sd.vae.eval()
            self.sd.vae.to(self.device_torch)
        else:
            # offload it. Already cached
            self.sd.vae.to('cpu')
            flush()
        add_all_snr_to_noise_scheduler(self.sd.noise_scheduler, self.device_torch)

        # get empty prompt embeds
        self.unconditional_prompt = self.sd.encode_prompt(
            [self.negative_prompt] * self.train_config.batch_size,
            [self.negative_prompt] * self.train_config.batch_size,
            long_prompts=True
        ).to(
            self.device_torch,
            dtype=get_torch_dtype(self.train_config.dtype))

        if self.use_taesd:
            if self.model_config.is_xl:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            else:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            self.taesd.to(dtype=self.torch_dtype, device=self.device_torch)
            self.taesd.eval()
            self.taesd.requires_grad_(False)

        if self.style_weight > 0 or self.content_weight > 0:
            self.setup_vgg19()
            self.vgg_19.requires_grad_(False)
            self.vgg_19.eval()

        # if we are only using taesd, then replace normal vae
        if self.use_taesd and not self.use_vae:
            self.sd.vae = self.taesd
            self.sd.pipeline.vae = self.taesd

        # self.update_sample_mean_std()
        self.load_terms()
        self.sd.vae = self.sd.vae.to(self.device_torch, dtype=self.torch_dtype)

        flush()

    def load_terms(self):
        if self.terms_location is None:
            raise Exception("Terms location is not set")
        if not os.path.exists(self.terms_location):
            raise Exception(f"Terms location does not exist: {self.terms_location}")
        with open(self.terms_location, 'r') as f:
            self.terms_list = f.read().split('\n')
        self.terms_list = [t.strip() for t in self.terms_list if t.strip() != '']
        self.print(f"Loaded {len(self.terms_list)} terms from {self.terms_location}")

    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return batch

    def before_unet_predict(self):
        pass

    def after_unet_predict(self):
        pass

    def end_of_training_loop(self):
        pass

    def hook_train_loop(self, batch: Optional[DataLoaderBatchDTO]):
        self.sd.vae.eval()
        self.sd.vae.to(self.device_torch)

        # if self.num_turbo_steps == 1:
        #     num_turbo_steps = 1
        # else:
        #     num_turbo_steps = self.num_turbo_steps // 2 if self.step_num % 2 == 0 else 1
        num_turbo_steps = self.num_turbo_steps

        # since we are augmenting, set this lower
        #  self.train_config.min_denoising_steps = self.train_config.min_denoising_steps // 4
        #  self.train_config.max_denoising_steps = self.train_config.max_denoising_steps // 4
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)
            if batch is not None:
                # just load it normally
                # encode the latents
                batch.latents = self.sd.encode_images(
                    batch.tensor.to(self.device_torch, dtype=dtype),
                    device=self.device_torch,
                    dtype=dtype,
                )

                positive_prompts = batch.get_caption_list()
                negative_prompts = [''] * len(positive_prompts)
                self.network.multiplier = batch.get_network_weight_list()

            else:
                positive_prompts = []
                negative_prompts = []

                for i in range(self.train_config.batch_size):
                    positive_prompt_chunks = []
                    negative_prompt_chunks = []
                    # get random amount of chunks to make longer prompts
                    for j in range(random.randint(1, 5)):
                        positive_prompt_chunks.append(self.terms_list[random.randint(0, len(self.terms_list) - 1)])
                    positive_prompts.append(', '.join(positive_prompt_chunks))
                    for j in range(random.randint(1, 5)):
                        negative_prompt_chunks.append(self.terms_list[random.randint(0, len(self.terms_list) - 1)])
                    negative_prompts.append(', '.join(negative_prompt_chunks))

                # set the weights
                self.network.multiplier = [1.0 for _ in range(self.train_config.batch_size)]

            self.optimizer.zero_grad(set_to_none=True)

            with self.timer('encode_prompt'):
                with torch.set_grad_enabled(False):
                    # make sure it is in eval mode
                    if isinstance(self.sd.text_encoder, list):
                        for te in self.sd.text_encoder:
                            te.eval()
                    else:
                        self.sd.text_encoder.eval()

                    new_positive_prompts = []
                    for prompt in positive_prompts:
                        if self.embedding is not None:
                            prompt = self.embedding.inject_embedding_to_prompt(
                                prompt, add_if_not_present=False
                            )
                        if self.trigger_word is not None:
                            prompt = self.sd.inject_trigger_into_prompt(
                                prompt, self.trigger_word, add_if_not_present=False
                            )
                        new_positive_prompts.append(prompt)

                    positive_prompts = new_positive_prompts

                    new_negative_prompts = []
                    for prompt in negative_prompts:
                        if self.embedding is not None:
                            prompt = self.embedding.inject_embedding_to_prompt(
                                prompt, add_if_not_present=False
                            )
                        if self.trigger_word is not None:
                            prompt = self.sd.inject_trigger_into_prompt(
                                prompt, self.trigger_word, add_if_not_present=False
                            )
                        new_negative_prompts.append(prompt)

                    negative_prompts = new_negative_prompts


                do_te_train = self.train_config.train_text_encoder
                if do_te_train:
                    # if is a list
                    if isinstance(self.sd.text_encoder, list):
                        for te in self.sd.text_encoder:
                            te.train()
                    else:
                        self.sd.text_encoder.train()
                else:
                    if isinstance(self.sd.text_encoder, list):
                        for te in self.sd.text_encoder:
                            te.eval()
                    else:
                        self.sd.text_encoder.eval()
                with torch.set_grad_enabled(False):
                    # get positive and negative prompts
                    embeds = self.sd.encode_prompt(
                        positive_prompts + negative_prompts,
                        dropout_prob=self.train_config.prompt_dropout_prob,
                        long_prompts=False
                    ).to(
                        self.device_torch,
                        dtype=dtype
                    )
                if not do_te_train:
                    embeds = embeds.detach()

            # get midpoint timestep. random number between 1 and num_inference_steps - 1 (never get a 0 idx as we need some reference)
            if num_turbo_steps == 1:
                # adjust the value for the first 1000 steps so step 0 = 0.5, step 1000 = 0.1
                # top_range_step = min(self.step_num, 1000)
                # if batch is not None:
                #     midpoint_min_idx = 0.1
                #     midpoint_max_idx = 0.4
                # else:
                #     midpoint_min_idx = value_map(top_range_step, 0, 1000, 0.33, 0.1)
                #     midpoint_max_idx = 0.66
                #
                # midpoint_timestep_idx = random.randint(math.floor(self.num_inference_steps * midpoint_min_idx),
                #                                        math.floor(self.num_inference_steps * midpoint_max_idx))
                rand_float = random.random()
                # this will favor the lower numbers
                if self.train_config.content_or_style == 'content':
                    midpoint_timestep_idx = rand_float ** 3 * self.num_inference_steps
                elif self.train_config.content_or_style == 'style':
                    # this will favor the higher numbers
                    midpoint_timestep_idx = (1 - rand_float ** 3) * self.num_inference_steps
                else:
                    midpoint_timestep_idx = rand_float * self.num_inference_steps

                midpoint_timestep_idx = math.floor(midpoint_timestep_idx)
                if midpoint_timestep_idx == 0:
                    midpoint_timestep_idx = 1
                if midpoint_timestep_idx >= self.num_inference_steps:
                    midpoint_timestep_idx = self.num_inference_steps - 1

            else:
                midpoint_timestep_idx = num_turbo_steps // 2

            self.child_scheduler = self.sd.noise_scheduler
            self.sd.noise_scheduler = self.teacher_scheduler
            self.sd.noise_scheduler.set_timesteps(
                self.num_inference_steps,
                device=self.device_torch
            )
            # get noise
            if batch is not None:
                batch_size = batch.latents.shape[0]
                timesteps_idx = torch.tensor(
                    [midpoint_timestep_idx] * batch_size,
                    device=self.device_torch,
                    dtype=torch.long
                ).long()
                # timesteps = self.sd.noise_scheduler.timesteps[timesteps_idx:timesteps_idx + 1]
                timestep_chunks = timesteps_idx.chunk(batch_size, dim=0)
                timesteps = []
                for chunk in timestep_chunks:
                    timesteps.append(self.sd.noise_scheduler.timesteps[chunk[0]:chunk[0] + 1])
                timesteps = torch.cat(timesteps, dim=0)
                # timesteps = self.sd.noise_scheduler.timesteps[timesteps_idx:timesteps_idx + 1]
                # timesteps = self.sd.noise_scheduler.timesteps[-timesteps_idx:-(timesteps_idx + 1)]

                noise = self.sd.get_latent_noise(
                    height=batch.latents.shape[2],
                    width=batch.latents.shape[3],
                    batch_size=batch.latents.shape[0],
                    noise_offset=self.train_config.noise_offset
                ).to(self.device_torch, dtype=dtype).detach()

                # noise = noise * self.teacher_scheduler.init_noise_sigma
                # noise = self.sd.noise_scheduler.scale_model_input(noise, timesteps)

                # if self.teacher_scheduler_name.startswith('euler'):
                #     init_schduler_noise = self.teacher_scheduler.sigmas[midpoint_timestep_idx - 1]
                #     noise = noise * init_schduler_noise

                # timesteps = self.sd.noise_scheduler.timesteps[timesteps_idx - 1:timesteps_idx]

                batch_latents = batch.latents.to(self.device_torch, dtype=dtype).detach().clone()

                # is is a euler scheduler scale them up

                mid_point_latents = self.sd.add_noise(
                    batch_latents.to(self.device_torch, dtype=dtype),
                    noise,
                    timesteps
                ).detach().clone()

                mid_point_latents = mid_point_latents.detach().clone()



                target_latents = batch.latents.to(self.device_torch, dtype=dtype).detach().clone()

            else:
                noise = self.sd.get_latent_noise(
                    pixel_height=self.size,
                    pixel_width=self.size,
                    batch_size=self.train_config.batch_size,
                    noise_offset=self.train_config.noise_offset,
                ).to(self.device_torch, dtype=dtype)

                teacher_noise = noise * self.teacher_scheduler.init_noise_sigma
                teacher_noise = teacher_noise.to(self.device_torch, dtype=dtype).detach()

                self.sd.unet.eval()
                with self.timer('generate_target'):
                    assert not self.network.is_active
                    # todo make sure we are not repeating any
                    teacher_guidance_scale = 6
                    mid_point_latents = self.sd.diffuse_some_steps(
                        teacher_noise,  # pass simple noise latents
                        embeds.detach(),
                        start_timesteps=0,
                        total_timesteps=midpoint_timestep_idx,
                        guidance_scale=teacher_guidance_scale,
                    ).detach()
                    target_latents = self.sd.diffuse_some_steps(
                        mid_point_latents,  # pass simple noise latents
                        embeds.detach(),
                        start_timesteps=midpoint_timestep_idx,
                        total_timesteps=self.num_inference_steps,
                        guidance_scale=teacher_guidance_scale,
                    ).detach()

            self.sd.unet.train()
            self.before_unet_predict()

            torch.cuda.empty_cache()

        # grad begins
        with self.sd.network:
            # flush()
            with self.timer('predict_unet'):
                assert self.network.is_active
                if num_turbo_steps == 1:
                    self.sd.noise_scheduler.set_timesteps(1, device=self.device_torch)
                    # we need to create our own one step timestep since we are not doing a full denoise one step.
                    self.sd.noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device_torch)
                    # find our midpoint timestep value in the scheduler
                    midpoint_timestep = self.sd.noise_scheduler.timesteps[
                                        midpoint_timestep_idx:midpoint_timestep_idx + 1]
                    # since we are just doing one step, we just need this in an array
                    single_step_timestep_schedule = [midpoint_timestep.item()]
                    # we get the timestep from the trainer but set it in the child
                    self.sd.noise_scheduler = self.child_scheduler

                    # if doing a euler timestep we have to build them completly ourselves see if scheduler includes the word euler
                    # use # trailing timesteps
                    if self.train_config.noise_scheduler.startswith('euler'):
                        self.sd.noise_scheduler.set_timesteps(
                            1,
                            device=self.device_torch
                        )
                        # step_ratio = self.sd.noise_scheduler.config.num_train_timesteps / self.sd.noise_scheduler.num_inference_steps
                        step_ratio = self.sd.noise_scheduler.config.num_train_timesteps / self.num_inference_steps
                        # creates integer timesteps by multiplying by ratio
                        # casting to int to avoid issues when num_inference_step is power of 3
                        timesteps = (np.arange(self.sd.noise_scheduler.config.num_train_timesteps, 0,
                                               -step_ratio)).round().copy().astype(
                            np.float32)
                        timesteps -= 1
                        sigmas = np.array(((
                                                       1 - self.sd.noise_scheduler.alphas_cumprod) / self.sd.noise_scheduler.alphas_cumprod) ** 0.5)
                        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
                        # extract the sigma idx for our midpoint timestep
                        sigmas = sigmas[midpoint_timestep_idx:midpoint_timestep_idx + 1]
                        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
                        self.sd.noise_scheduler.sigmas = torch.from_numpy(sigmas).to(device=self.device_torch)
                        self.sd.noise_scheduler.timesteps = torch.from_numpy(
                            np.array(single_step_timestep_schedule, dtype=np.float32)).to(device=self.device_torch)
                    else:
                        self.sd.noise_scheduler.set_timesteps(
                            device=self.device_torch,
                            timesteps=single_step_timestep_schedule
                        )
                    start_step = 0
                else:
                    self.sd.noise_scheduler = self.child_scheduler
                    # we can just do it as normal for now. Not ideal but probably only doing 1 step
                    self.sd.noise_scheduler.set_timesteps(num_turbo_steps, device=self.device_torch)
                    # step_scaler = num_turbo_steps / self.num_inference_steps
                    start_step = num_turbo_steps // 2
                # if ddpm we need to do this for some reason
                if self.train_config.noise_scheduler in ['ddpm']:
                    try:
                        self.sd.noise_scheduler.betas = self.sd.noise_scheduler.betas.to(self.device_torch)
                        self.sd.noise_scheduler.alphas = self.sd.noise_scheduler.alphas.to(self.device_torch)
                        self.sd.noise_scheduler.alphas_cumprod = self.sd.noise_scheduler.alphas_cumprod.to(
                            self.device_torch)
                    except Exception as e:
                        pass



                latents = self.sd.diffuse_some_steps(
                    mid_point_latents,  # pass simple noise latents
                    embeds,
                    start_timesteps=start_step,
                    total_timesteps=num_turbo_steps,
                    guidance_scale=self.guidance_scale,
                    # is_input_scaled=batch is not None
                    is_input_scaled=False
                    # is_input_scaled=batch is not None
                )

                latents = latents.to(self.device_torch, dtype=dtype)

            # randomly choose a decoder
            # TAESD has normalizers on it  that can knock off the scaling factor
            # we alternate back and forth between vaes so it can learn both and prevent mode collapse
            # decoders = [self.sd.vae, self.taesd]
            # decoders = [self.taesd]
            decoders = []
            if self.use_taesd:
                decoders.append(self.taesd)
            if self.use_vae:
                decoders.append(self.sd.vae)
            decoder = decoders[math.floor(random.random() * len(decoders))]

            if batch is not None:
                latents_scaled = latents / decoder.config['scaling_factor']
                img_pred = decoder.decode(latents_scaled).sample
                # img_pred = img_pred / decoder.config['scaling_factor']
                img_target = batch.tensor.to(self.device_torch, dtype=dtype).detach()

            else:
                combined_latents = torch.cat([latents, target_latents], dim=0).to(self.device_torch, dtype=dtype)

                combined_latents = combined_latents / decoder.config['scaling_factor']
                # TAESD wont checkpoint. But we can checkpoint the others
                # if self.train_config.gradient_checkpointing and not decoder == self.taesd:
                #     # combined_imgs = checkpoint(decoder.decode, combined_latents).sample
                #     combined_imgs = checkpoint(decoder.decode, combined_latents).sample
                # else:
                combined_imgs = decoder.decode(combined_latents).sample

                # add pixel mask
                img_pred, img_target = torch.chunk(combined_imgs, 2, dim=0)

            img_pred_show = img_pred.clone()
            img_target_show = img_target.clone()

            self.after_unet_predict()

            with self.timer('calculate_loss'):
                stacked = torch.cat([img_pred, img_target], dim=0)
                # stacked = torch.cat([img_pred, torch.zeros_like(imgs)], dim=0)
                stacked = (stacked / 2 + 0.5).clamp(0, 1)
                # make them smaller to prevent nans
                stacked = stacked * 0.5
                self.vgg_19(stacked)
                if torch.isnan(self.vgg19_pool_4.tensor).any():
                    raise ValueError('vgg19_pool_4 has nan values')

                # loss_l2 = torch.nn.functional.mse_loss(
                #     latents.float(),
                #     target_latents.float()
                #     # torch.zeros_like(batch.latents.float())
                # ) * self.l2_weight
                loss_l2 = torch.nn.functional.mse_loss(
                    img_pred.float(),
                    img_target.float()
                    # torch.zeros_like(batch.latents.float())
                ) * self.l2_weight
                # Run through VGG19
                if self.style_weight > 0 or self.content_weight > 0:

                    style_loss = self.get_style_loss() * self.style_weight
                    content_loss = self.get_content_loss() * self.content_weight
                else:
                    style_loss = torch.tensor(0.0, device=self.device_torch)
                    content_loss = torch.tensor(0.0, device=self.device_torch)

                # if self.use_critic:
                #     # Critic Loss
                #     critic_d_loss = self.critic.step(self.vgg19_pool_4.tensor.detach())
                #     if self.step_num < self.critic_warmup_steps:
                #         critic_gen_loss = torch.tensor(0.0, device=self.device_torch)
                #     else:
                #         critic_gen_loss = self.critic.get_critic_loss(self.vgg19_pool_4.tensor) * self.critic_weight
                # else:
                #     critic_gen_loss = torch.tensor(0.0, device=self.device_torch)
                #     critic_d_loss = torch.tensor(0.0, device=self.device_torch)

            # loss = loss_l2 + style_loss + content_loss + critic_gen_loss
            loss = loss_l2 + style_loss + content_loss

            # check if nan
            if torch.isnan(loss):
                raise ValueError("loss is nan")

            # run pure noise pred through vgg19 and critic

            with self.timer('backward'):
                # IMPORTANT if gradient checkpointing do not leave with network when doing backward
                # it will destroy the gradients. This is because the network is a context manager
                # and will change the multipliers back to 0.0 when exiting. They will be
                # 0.0 for the backward pass and the gradients will be 0.0
                # I spent weeks on fighting this. DON'T DO IT
                loss.backward()

            pure_img_pred_show = None
            # if self.use_critic:
            # with self.timer('single_run'):
            #     self.sd.noise_scheduler.set_timesteps(1, device=self.device_torch)
            #     scaled_noise = noise * self.sd.noise_scheduler.init_noise_sigma
            #     scaled_noise = scaled_noise.to(self.device_torch, dtype=dtype).detach()
            #
            #     pure_pred = self.sd.diffuse_some_steps(
            #         scaled_noise.detach().clone().to(self.device_torch, dtype=dtype),
            #         embeds,
            #         start_timesteps=0,
            #         total_timesteps=1,
            #         guidance_scale=self.guidance_scale,
            #         is_input_scaled=False
            #     ).to(self.device_torch, dtype=dtype)
            #
            #     pure_pred = pure_pred / decoder.config['scaling_factor']
            #     pure_img_pred = decoder.decode(pure_pred).sample
            #     pure_img_pred_show = pure_img_pred.clone().detach()
            #
            #
            #     stacked = torch.cat([pure_img_pred, img_target.detach()], dim=0)
            #     # stacked = torch.cat([img_pred, torch.zeros_like(imgs)], dim=0)
            #     stacked = (stacked / 2 + 0.5).clamp(0, 1)
            #     # make them smaller to prevent nans
            #     stacked = stacked * 0.5
            #     self.vgg_19(stacked)
            #     if torch.isnan(self.vgg19_pool_4.tensor).any():
            #         raise ValueError('vgg19_pool_4 has nan values')
            #
            #     # do the critic step completely separately since we want a full image generation without bleed
            #     # Critic Loss
            #     if self.step_num < self.critic_warmup_steps or not self.use_critic:
            #         critic_d_loss = torch.tensor(0.0, device=self.device_torch)
            #         critic_gen_loss = torch.tensor(0.0, device=self.device_torch)
            #         additional_style_loss = self.get_style_loss() * self.style_weight
            #         additional_loss_l2 = torch.tensor(0.0, device=self.device_torch)
            #         # additional_loss_l2 = torch.nn.functional.mse_loss(
            #         #     pure_img_pred.float(),
            #         #     img_target.detach().clone().float()
            #         #     # torch.zeros_like(batch.latents.float())
            #         # ) * self.l2_weight
            #         # additional_content_loss = self.get_content_loss() * self.content_weight * 0.5
            #         additional_content_loss = torch.tensor(0.0, device=self.device_torch)
            #         additional_loss = additional_style_loss + additional_content_loss + additional_loss_l2
            #         additional_loss.backward()
            #
            #     else:
            #         critic_d_loss = self.critic.step(self.vgg19_pool_4.tensor.detach())
            #         critic_gen_loss = self.critic.get_critic_loss(self.vgg19_pool_4.tensor) * self.critic_weight
            #         additional_style_loss = self.get_style_loss() * self.style_weight
            #         # additional_loss_l2 = torch.nn.functional.mse_loss(
            #         #     pure_img_pred.float(),
            #         #     img_target.detach().clone().float()
            #         #     # torch.zeros_like(batch.latents.float())
            #         # ) * self.l2_weight
            #         additional_loss_l2 = torch.tensor(0.0, device=self.device_torch)
            #         # additional_content_loss = self.get_content_loss() * self.content_weight * 0.5
            #         additional_content_loss = torch.tensor(0.0, device=self.device_torch)
            #         additional_loss = additional_style_loss + additional_content_loss + additional_loss_l2
            #         critic_gen_loss = additional_loss + critic_gen_loss
            #         critic_gen_loss.backward()
            #
            #     additional_style_loss = additional_style_loss.detach().clone()
            #     style_loss = style_loss.detach().clone()
            #     style_loss = style_loss + additional_style_loss
            #     additional_content_loss = additional_content_loss.detach().clone()
            #     content_loss = content_loss.detach().clone()
            #     content_loss = content_loss + additional_content_loss


            with torch.no_grad():
                with self.timer('show_img'):
                    # todo make this optional
                    # determine how to make a grid based on 2 images per batch size

                    if img_pred_show.shape[0] % 2 == 0:
                        # split and stack height
                        img_pred_show = torch.cat(torch.split(img_pred_show, 2, dim=0), dim=2)
                        img_target_show = torch.cat(torch.split(img_target_show, 2, dim=0), dim=2)
                        if pure_img_pred_show is not None:
                            pure_img_pred_show = torch.cat(torch.split(pure_img_pred_show, 2, dim=0), dim=2)

                    if img_pred_show.shape[0] % 2 == 0:
                        # do it again
                        img_pred_show = torch.cat(torch.split(img_pred_show, 2, dim=0), dim=3)
                        img_target_show = torch.cat(torch.split(img_target_show, 2, dim=0), dim=3)
                        if pure_img_pred_show is not None:
                            pure_img_pred_show = torch.cat(torch.split(pure_img_pred_show, 2, dim=0), dim=3)

                    # but img_pred next to imgs
                    if pure_img_pred_show is not None:
                        show_tensors(torch.cat([pure_img_pred_show, img_pred_show, img_target_show], dim=3), name=self.name)
                    else:
                        show_tensors(torch.cat([img_pred_show, img_target_show], dim=3), name=self.name)

            if not self.is_grad_accumulation_step:
                torch.nn.utils.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
                # only step if we are not accumulating
                with self.timer('optimizer_step'):
                    # apply gradients
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                # gradient accumulation. Just a place for breakpoint
                pass

        # TODO Should we only step scheduler on grad step? If so, need to recalculate last step
        with self.timer('scheduler_step'):
            self.lr_scheduler.step()

        loss_dict = OrderedDict([
            ('mse', loss_l2),
            ('sty', style_loss),
            ('con', content_loss),
            ('loss', loss),
        ])
        # if self.use_critic:
        #     loss_dict['critic_d'] = critic_d_loss
        #     loss_dict['critic_g'] = critic_gen_loss

        if self.train_config.noise_scheduler == 'custom_lcm':
            self.sd.noise_scheduler.original_inference_steps = 50

        self.end_of_training_loop()

        return loss_dict
