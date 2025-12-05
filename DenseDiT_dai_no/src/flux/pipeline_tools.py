from diffusers.pipelines import FluxPipeline, FluxKontextPipeline
from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger
from torch import Tensor


def encode_images(pipeline: FluxKontextPipeline, images: Tensor, device=None, dtype=None):
    if device is None:
        device = pipeline.device
    if dtype is None:
        dtype = pipeline.dtype

    images = images.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
    images = pipeline.image_processor.preprocess(images)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        device,
        dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            device,
            dtype,
        )
        
    return images_tokens.to(dtype=dtype, device=device), images_ids.to(dtype=dtype, device=device)


def prepare_text_input(pipeline: FluxKontextPipeline, prompts, max_sequence_length=512, device=None, dtype=None):
    if device is None:
        device = pipeline.device
    if dtype is None:
        dtype = pipeline.dtype
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    prompt_embeds = prompt_embeds.to(device=device,dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device,dtype=dtype)
    text_ids = text_ids.to(device=device,dtype=dtype)
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids
