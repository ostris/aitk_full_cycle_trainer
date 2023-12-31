import math
import os
from collections import OrderedDict
import random
from typing import Union
from diffusers import T2IAdapter, AutoencoderTiny
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import v2

from toolkit import train_tools
from toolkit.basic import get_mean_std
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.image_utils import show_tensors
from PIL import Image
from toolkit.prompt_utils import PromptEmbeds
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


class SDFullCycleTrainer(BaseSDTrainProcess):

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
        self.style_weight = self.config.get('style_weight', 1e0)
        self.content_weight = self.config.get('content_weight', 1e-1)
        self.l2_weight = self.config.get('l2_weight', 1e-2)
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
            size = self.datasets[0].resolution
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

        self.update_sample_mean_std()
        flush()

    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return batch

    def before_unet_predict(self):
        pass

    def after_unet_predict(self):
        pass

    def end_of_training_loop(self):
        pass

    def apply_inverse_standardization(self, tensor: torch.Tensor, scale_factor: float = 1.0):
        input_tensor = tensor
        # cache current mean_std for tensors
        t = torch.chunk(tensor, tensor.shape[0], dim=0)
        for chunk in t:
            mean, std = get_mean_std(chunk)
            if len(self.mean_std_cache) > self.max_cache_size:
                self.mean_std_cache.pop(0)
            self.mean_std_cache.append((mean, std))

        # get ave mean_std from cache
        avg_mean = 0
        avg_std = 0
        for m, s in self.mean_std_cache:
            avg_mean += m
            avg_std += s
        avg_mean = avg_mean / len(self.mean_std_cache)
        avg_std = avg_std / len(self.mean_std_cache)

        if self.sample_mean_std is not None:
            sample_mean, sample_std = self.sample_mean_std
            # Adjusting normalization with scale factor
            tensor = (tensor - sample_mean) / (sample_std * scale_factor)
            # Adjusting denormalization with scale factor
            tensor = tensor * (avg_std * scale_factor) + avg_mean
        # if self.sample_mean_std is not None:
        #     sample_mean, sample_std = self.sample_mean_std
        #     # Adjusting normalization with scale factor
        #     tensor = (tensor - avg_mean) / (avg_std * scale_factor)
        #     # Adjusting denormalization with scale factor
        #     tensor = tensor * (sample_std * scale_factor) + avg_std



        return (tensor * 0.5) + (input_tensor * 0.5)

    def augment_tensor(self, tensor: torch.Tensor):
        # tensor = tensor * 0.5 + 0.5
        # elastic_transformer = v2.ElasticTransform(alpha=250.0)
        # jitter = v2.ColorJitter(brightness=.5, hue=.3)
        # blurrer = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
        # # equalizer = v2.RandomEqualize()
        #
        # # 1/4 the time, convert to grayscale
        # if random.random() < 0.50:
        #     tensor = v2.Grayscale(num_output_channels=3)(tensor)
        #
        # transforms_list = [elastic_transformer, jitter, blurrer]
        # # shuffle
        # random.shuffle(transforms_list)
        #
        # for transform in transforms_list:
        #     tensor = transform(tensor)
        #
        # tensor = tensor * 2.0 - 1.0

        # tensor = self.apply_inverse_standardization(tensor, 1.0)

        # clip
        tensor = torch.clamp(tensor, -1, 1)

        return tensor


    def hook_train_loop(self, batch: 'DataLoaderBatchDTO'):
        if self.steps_in_cycle == 4 or self.steps_in_cycle == 2:
            self.train_config.min_denoising_steps = 450
            self.train_config.max_denoising_steps = 650
            start_step = 2
            if self.steps_in_cycle == 2:
                start_step = 1
        elif self.steps_in_cycle == 3:
            self.train_config.min_denoising_steps = 200
            self.train_config.max_denoising_steps = 450
            start_step = 2
        else:
            raise ValueError(f"Steps in cycle {self.steps_in_cycle} not supported. Try 2, 3, or 4")

       # since we are augmenting, set this lower
       #  self.train_config.min_denoising_steps = self.train_config.min_denoising_steps // 4
       #  self.train_config.max_denoising_steps = self.train_config.max_denoising_steps // 4

        self.timer.start('preprocess_batch')
        batch = self.preprocess_batch(batch)
        dtype = get_torch_dtype(self.train_config.dtype)
        # we need to keep our target image and aument the source image before preprocessing
        target_imgs = batch.tensor.clone().to(self.device_torch, dtype=self.torch_dtype)
        augmented_imgs = self.augment_tensor(target_imgs.to(self.device_torch, dtype=torch.float32))
        batch.tensor = (augmented_imgs.to(self.device_torch, dtype=self.torch_dtype) * 0.5) + (target_imgs * 0.5)

        noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)
        # replace the targets not that it is all encoded
        imgs = target_imgs
        batch.tensor = imgs

        network_weight_list = batch.get_network_weight_list()

        self.timer.stop('preprocess_batch')
        # flush()
        with self.timer('grad_setup'):

            # text encoding
            grad_on_text_encoder = False
            if self.train_config.train_text_encoder:
                grad_on_text_encoder = True

            if self.embedding:
                grad_on_text_encoder = True

            # have a blank network so we can wrap it in a context and set multipliers without checking every time
            if self.network is not None:
                network = self.network
            else:
                network = BlankNetwork()

            # set the weights
            network.multiplier = network_weight_list
            self.optimizer.zero_grad(set_to_none=True)

        with network:
            with self.timer('encode_prompt'):
                if grad_on_text_encoder:
                    with torch.set_grad_enabled(True):
                        conditional_embeds = self.sd.encode_prompt(
                            conditioned_prompts,
                            dropout_prob=self.train_config.prompt_dropout_prob,
                            long_prompts=True).to(
                            self.device_torch,
                            dtype=dtype)
                else:
                    with torch.set_grad_enabled(False):
                        # make sure it is in eval mode
                        if isinstance(self.sd.text_encoder, list):
                            for te in self.sd.text_encoder:
                                te.eval()
                        else:
                            self.sd.text_encoder.eval()
                        conditional_embeds = self.sd.encode_prompt(
                            conditioned_prompts,
                            dropout_prob=self.train_config.prompt_dropout_prob,
                            long_prompts=True).to(
                            self.device_torch,
                            dtype=dtype)

                    # detach the embeddings
                    conditional_embeds = conditional_embeds.detach()

            self.before_unet_predict()

            with self.timer('predict_unet'):
                self.sd.noise_scheduler.set_timesteps(self.steps_in_cycle, device=self.device_torch)

                # randomly choose a decoder
                # TAESD has normalizers on it  that can knock off the scaling factor
                # we alternate back and forth between vaes so it can learn both and prevent mode collapse
                # decoders = [self.sd.vae, self.taesd]
                # decoders = [self.taesd]
                decoders = [self.sd.vae]
                decoder = decoders[math.floor(random.random() * len(decoders))]

                pred = self.sd.diffuse_some_steps(
                    noisy_latents,  # pass simple noise latents
                    train_tools.concat_prompt_embeddings(
                        self.unconditional_prompt,  # unconditional
                        conditional_embeds,  # target
                        1,
                    ),
                    start_timesteps=start_step,
                    total_timesteps=self.steps_in_cycle,
                    guidance_scale=1.0,
                )
                latent_pred = pred.to(dtype=get_torch_dtype(self.train_config.dtype))
                latent_pred = latent_pred / decoder.config['scaling_factor']
                # TAESD wont checkpoint. But we can checkpoint the others
                if self.train_config.gradient_checkpointing and not decoder == self.taesd:
                    img_pred = checkpoint(decoder.decode, latent_pred).sample
                else:
                    img_pred = decoder.decode(latent_pred).sample

                # add pixel mask
                img_pred = img_pred
                imgs = imgs

                with torch.no_grad():
                    # todo remove me
                    # show_latents(torch.cat([latent_pred, latent], dim=0), self.taesd)
                    # but img_pred next to imgs
                    show_tensors(torch.cat([img_pred, imgs], dim=3), name=self.name)
            self.after_unet_predict()

            with self.timer('calculate_loss'):
                # loss = torch.nn.functional.mse_loss(latent_pred, latent)
                # L2 Loss
                loss_l2 = torch.nn.functional.mse_loss(
                    pred.float(),
                    batch.latents.float()
                ) * self.l2_weight
                # Run through VGG19
                if self.style_weight > 0 or self.content_weight > 0:
                    stacked = torch.cat([img_pred, imgs], dim=0)
                    stacked = (stacked / 2 + 0.5).clamp(0, 1)
                    self.vgg_19(stacked)
                    if torch.isnan(self.vgg19_pool_4.tensor).any():
                        raise ValueError('vgg19_pool_4 has nan values')

                    style_loss = self.get_style_loss() * self.style_weight
                    content_loss = self.get_content_loss() * self.content_weight
                else:
                    style_loss = torch.tensor(0.0, device=self.device_torch)
                    content_loss = torch.tensor(0.0, device=self.device_torch)

                loss = loss_l2 + style_loss + content_loss

            # check if nan
            if torch.isnan(loss):
                raise ValueError("loss is nan")

            with self.timer('backward'):
                # IMPORTANT if gradient checkpointing do not leave with network when doing backward
                # it will destroy the gradients. This is because the network is a context manager
                # and will change the multipliers back to 0.0 when exiting. They will be
                # 0.0 for the backward pass and the gradients will be 0.0
                # I spent weeks on fighting this. DON'T DO IT
                loss.backward()

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

        self.end_of_training_loop()

        return loss_dict
