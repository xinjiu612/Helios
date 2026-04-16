import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from accelerate import Accelerator
from helios.dataset.dataloader_mp4_dist import (
    BucketedFeatureDataset,
    BucketedSampler,
    collate_fn,
    find_nearest_length_bucket,
    find_nearest_resolution_bucket,
)
from helios.utils.utils_base import encode_prompt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AutoencoderKLWan
from diffusers.training_utils import free_memory


def setup_distributed_env():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def cleanup_distributed_env():
    dist.destroy_process_group()


def maybe_sync_cuda(enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def log_timing(rank, step, timing_stats, extra_stats=None):
    stats = timing_stats.copy()
    if extra_stats is not None:
        stats.update(extra_stats)
    stats_str = " ".join(f"{key}={value}" for key, value in stats.items())
    print(f"[rank{rank}] step={step} {stats_str}", flush=True)


def async_copy_to_cpu(tensor, copy_stream):
    cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
    with torch.cuda.stream(copy_stream):
        cpu_tensor.copy_(tensor, non_blocking=True)
    return cpu_tensor


def save_batch_payloads(batch_payloads, copy_event=None, retained_gpu_tensors=None):
    if copy_event is not None:
        copy_event.synchronize()
    del retained_gpu_tensors

    batch_save_start = time.perf_counter()
    output_paths = []
    for payload in batch_payloads:
        temp_to_save = {
            "vae_latent": payload["vae_latent"],
            "image_latents": payload["image_latents"],
            "fake_image_latents": payload["fake_image_latents"],
            "prompt_embed": payload["prompt_embed"],
            "first_frames_image": transforms.ToPILImage()(payload["first_frames_image"]),
            "prompt_raw": payload["prompt_raw"],
        }
        torch.save(temp_to_save, payload["output_path"])
        output_paths.append(payload["output_path"])
    return output_paths, time.perf_counter() - batch_save_start, len(batch_payloads)


def drain_save_futures(save_futures, wait_for_one=False):
    if not save_futures:
        return [], 0.0, 0

    if wait_for_one:
        done, not_done = wait(save_futures, return_when=FIRST_COMPLETED)
        completed = list(done)
        remaining = list(not_done)
    else:
        completed = [future for future in save_futures if future.done()]
        remaining = [future for future in save_futures if not future.done()]

    completed_save_time = 0.0
    completed_save_count = 0
    for future in completed:
        try:
            output_paths, save_time, saved_count = future.result()
            completed_save_time += save_time
            completed_save_count += saved_count
            for output_path in output_paths:
                print(f"save latent to: {output_path}")
        except Exception as exc:
            print(f"Save task failed: {exc}", flush=True)

    return remaining, completed_save_time, completed_save_count


class DebugBucketedFeatureDataset(BucketedFeatureDataset):
    def __init__(
        self,
        *args,
        skip_video_scan=False,
        subset_size=None,
        subset_seed=42,
        subset_mode="head",
        **kwargs,
    ):
        self.skip_video_scan = skip_video_scan
        self.subset_size = subset_size
        self.subset_seed = subset_seed
        self.subset_mode = subset_mode
        super().__init__(*args, **kwargs)

    def _use_direct_metadata_mode(self):
        return self.skip_video_scan or self.subset_size is not None

    def _process_json_file(self, json_file, video_folder, cache_file):
        if not self._use_direct_metadata_mode():
            return super()._process_json_file(json_file, video_folder, cache_file)

        print(f"Debug direct metadata mode for file: {json_file}")
        print(f"  Video folder: {video_folder}")
        file_samples, file_buckets = self._build_file_metadata(json_file, video_folder)
        print(f"Prepared {len(file_samples)} samples without folder scan/cache from {json_file}\n")

        sample_idx_offset = len(self.samples)
        self.samples.extend(file_samples)
        for bucket_key, indices in file_buckets.items():
            adjusted_indices = [idx + sample_idx_offset for idx in indices]
            self.buckets[bucket_key].extend(adjusted_indices)

    @staticmethod
    def _normalize_video_file(path_value):
        return (
            path_value.replace("videos_clip_v1_20241111/", "")
            .replace("videos_clip_v2_20241111/", "")
            .replace("videos_clip_v4_20241111/", "")
        )

    def _select_records(self, data, json_file):
        total_records = len(data)
        if self.subset_size is None or self.subset_size >= total_records:
            selected_records = list(data)
        elif self.subset_mode == "random":
            rng = random.Random(self.subset_seed)
            selected_indices = sorted(rng.sample(range(total_records), self.subset_size))
            selected_records = [data[idx] for idx in selected_indices]
        else:
            selected_records = data[: self.subset_size]

        if self.subset_size is not None:
            print(
                f"Using subset of {len(selected_records)}/{total_records} records from {json_file} "
                f"(mode={self.subset_mode}, seed={self.subset_seed})"
            )
        return selected_records

    def _build_file_metadata(self, json_file, video_folder):
        if not self._use_direct_metadata_mode():
            return super()._build_file_metadata(json_file, video_folder)

        with open(json_file, "r") as f:
            data = json.load(f)

        selected_records = self._select_records(data, json_file)
        if self.skip_video_scan:
            print(f"Skipping full folder scan: {video_folder}")
        else:
            print(f"Bypassing full folder scan via subset-only mode: {video_folder}")

        samples = []
        buckets = defaultdict(list)
        sample_idx = 0

        print(f"Processing {len(selected_records)} records from {json_file} with stride={self.stride}...")
        for i, item in enumerate(selected_records):
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(selected_records)} records")

            video_file = self._normalize_video_file(item["path"])
            video_path = os.path.join(video_folder, video_file)
            if not os.path.isfile(video_path):
                print(f"missing video: {video_path}")
                continue

            cut_start_frame = item["cut"][0]
            cut_end_frame = item["cut"][1]
            num_frame = cut_end_frame - cut_start_frame

            if self.single_length:
                if num_frame < self.single_num_frame:
                    continue
            else:
                if num_frame < 33:
                    continue

            uttid = os.path.basename(video_file).replace(".mp4", "") + f"_{cut_start_frame}-{cut_end_frame}"
            fps = item["fps"]

            crop = item["crop"]
            width = crop[1] - crop[0]
            height = crop[3] - crop[2]
            prompt = item["cap"][0]

            effective_num_frame = (num_frame + self.stride - 1) // self.stride
            bucket_num_frame = find_nearest_length_bucket(effective_num_frame, stride=self.stride)
            bucket_height, bucket_width = find_nearest_resolution_bucket(height, width, resolution=self.resolution)

            if self.single_res or self.multi_res:
                allowed_resolutions = [(self.single_height, self.single_width)]
                if self.multi_res:
                    allowed_resolutions.extend(
                        [
                            (self.single_height // 2, self.single_width // 2),
                            (self.single_height // 4, self.single_width // 4),
                        ]
                    )
                if (bucket_height, bucket_width) not in allowed_resolutions:
                    continue
                bucket_height, bucket_width = random.choice(allowed_resolutions)

            if self.single_length:
                bucket_num_frame = self.single_num_frame

            if self.base_fps is not None:
                stride = max(int(fps / self.base_fps), 1)
                required_frames = bucket_num_frame * stride
                if required_frames >= num_frame:
                    continue
            else:
                stride = self.stride

            bucket_key = (bucket_num_frame, bucket_height, bucket_width)

            sample_info = {
                "uttid": uttid,
                "dataset_name": json_file.rstrip("/"),
                "video_folder": video_folder,
                "video_path": video_path,
                "bucket_key": bucket_key,
                "prompt": self.id_token + prompt,
                "fps": fps,
                "stride": stride,
                "effective_num_frame": effective_num_frame,
                "num_frame": num_frame,
                "height": height,
                "width": width,
                "bucket_num_frame": bucket_num_frame,
                "bucket_height": bucket_height,
                "bucket_width": bucket_width,
                "cut_start_frame": cut_start_frame,
                "cut_end_frame": cut_end_frame,
                "crop": crop,
            }

            samples.append(sample_info)
            buckets[bucket_key].append(sample_idx)
            sample_idx += 1

        return samples, buckets


def main(
    rank,
    world_size,
    global_rank,
    stride,
    batch_size,
    dataloader_num_workers,
    dataloader_prefetch_factor,
    dataloader_persistent_workers,
    debug_timing,
    timing_every,
    timing_sync_cuda,
    save_workers,
    max_pending_saves,
    json_file,
    video_folder,
    output_latent_folder,
    pretrained_model_name_or_path,
    resolution=640,
    skip_video_scan=False,
    subset_size=None,
    subset_seed=42,
    subset_mode="head",
    sync_save=False,
):
    weight_dtype = torch.bfloat16
    device = rank
    seed = 42
    save_executor = None if sync_save else ThreadPoolExecutor(max_workers=save_workers)
    save_futures = []
    copy_stream = None if sync_save else torch.cuda.Stream(device=device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device, weight_dtype
    )

    vae.eval()
    vae.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    dataset = DebugBucketedFeatureDataset(
        json_files=json_file,
        video_folders=video_folder,
        stride=stride,
        force_rebuild=False,
        resolution=resolution,
        single_res=True,
        single_height=384,
        single_width=640,
        skip_video_scan=skip_video_scan,
        subset_size=subset_size,
        subset_seed=subset_seed,
        subset_mode=subset_mode,
    )
    sampler = BucketedSampler(dataset, batch_size=batch_size, drop_last=True, shuffle=True, seed=seed)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        persistent_workers=dataloader_persistent_workers and dataloader_num_workers > 0,
        prefetch_factor=dataloader_prefetch_factor if dataloader_num_workers > 0 else None,
    )

    print(len(dataset), len(dataloader))
    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)
    print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")
    print(f"Process index: {accelerator.process_index}, World size: {accelerator.num_processes}")

    sampler.set_epoch(0)
    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="Processing")

    data_iter = iter(dataloader)
    idx = 0
    gc_interval = 50
    try:
        while True:
            wait_batch_start = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            wait_batch_time = time.perf_counter() - wait_batch_start

            if batch is None or batch["videos"] is None:
                print("None batch, continuing")
                idx += 1
                continue

            raw_batch_size = 0 if batch["uttid"] is None else len(batch["uttid"])
            free_memory_before_time = 0.0

            valid_indices = []
            valid_uttids = []
            valid_num_frames = []
            valid_heights = []
            valid_widths = []
            valid_videos = []
            valid_prompts = []
            valid_first_frames_images = []

            if batch["uttid"] is None:
                print("None batch, contiuning")
                idx += 1
                continue

            filter_start = time.perf_counter()
            for i, (uttid, num_frame, height, width) in enumerate(
                zip(
                    batch["uttid"],
                    batch["video_metadata"]["num_frames"],
                    batch["video_metadata"]["height"],
                    batch["video_metadata"]["width"],
                )
            ):
                os.makedirs(output_latent_folder, exist_ok=True)
                output_path = os.path.join(output_latent_folder, f"{uttid}_{num_frame}_{height}_{width}.pt")
                if not os.path.exists(output_path):
                    valid_indices.append(i)
                    valid_uttids.append(uttid)
                    valid_num_frames.append(num_frame)
                    valid_heights.append(height)
                    valid_widths.append(width)
                    valid_videos.append(batch["videos"][i])
                    valid_prompts.append(batch["prompts"][i])
                    valid_first_frames_images.append(batch["first_frames_images"][i])
                else:
                    print(f"skipping {uttid}")
            filter_time = time.perf_counter() - filter_start

            if not valid_indices:
                print("skipping entire batch!")
                if rank == 0:
                    pbar.update(1)
                    pbar.set_postfix({"batch": idx})
                if debug_timing and idx % timing_every == 0:
                    log_timing(
                        global_rank,
                        idx,
                        {
                            "wait_batch_s": f"{wait_batch_time:.4f}",
                            "filter_s": f"{filter_time:.4f}",
                            "free_mem_pre_s": f"{free_memory_before_time:.4f}",
                        },
                        {
                            "raw_batch": raw_batch_size,
                            "valid_batch": 0,
                            "skipped": True,
                            "pending_saves": len(save_futures),
                        },
                    )
                idx += 1
                continue

            batch = None
            del batch
            free_memory_mid_time = 0.0

            rebuild_start = time.perf_counter()
            batch = {
                "uttid": valid_uttids,
                "video_metadata": {"num_frames": valid_num_frames, "height": valid_heights, "width": valid_widths},
                "videos": torch.stack(valid_videos),
                "prompts": valid_prompts,
                "first_frames_images": torch.stack(valid_first_frames_images),
            }
            rebuild_time = time.perf_counter() - rebuild_start

            if len(batch["uttid"]) == 0:
                print("All samples in this batch are already processed, skipping!")
                idx += 1
                continue

            valid_batch_size = len(batch["uttid"])
            num_frames = batch["video_metadata"]["num_frames"][0]

            with torch.no_grad():
                maybe_sync_cuda(timing_sync_cuda)
                h2d_start = time.perf_counter()
                pixel_values = batch["videos"].permute(0, 2, 1, 3, 4).to(dtype=vae.dtype, device=device, non_blocking=True)
                maybe_sync_cuda(timing_sync_cuda)
                h2d_time = time.perf_counter() - h2d_start

                latent_window_size = 9
                frame_window_size = (latent_window_size - 1) * 4 + 1
                num_latent_frames = pixel_values.shape[2]
                num_chunk_to_encode = num_latent_frames // frame_window_size

                history_latent_list = []
                maybe_sync_cuda(timing_sync_cuda)
                vae_encode_start = time.perf_counter()
                for i in range(num_chunk_to_encode):
                    start_idx = i * frame_window_size
                    end_idx = start_idx + frame_window_size
                    cur_pixel_values = pixel_values[:, :, start_idx:end_idx, :, :]
                    cur_latent = vae.encode(cur_pixel_values).latent_dist.sample()
                    cur_latent = (cur_latent - latents_mean) * latents_std
                    history_latent_list.append(cur_latent)
                vae_latents = torch.stack(history_latent_list, dim=1)
                maybe_sync_cuda(timing_sync_cuda)
                vae_encode_time = time.perf_counter() - vae_encode_start

                maybe_sync_cuda(timing_sync_cuda)
                image_encode_start = time.perf_counter()
                image_pixel_values = pixel_values[:, :, :1, :, :]
                image_latents = vae.encode(image_pixel_values).latent_dist.sample()
                image_latents = (image_latents - latents_mean) * latents_std

                fake_video = image_pixel_values.repeat(1, 1, frame_window_size, 1, 1)
                fake_image_latents = vae.encode(fake_video).latent_dist.sample()
                fake_image_latents = (fake_image_latents - latents_mean) * latents_std
                fake_image_latents = fake_image_latents[:, :, -1:, :, :]
                maybe_sync_cuda(timing_sync_cuda)
                image_encode_time = time.perf_counter() - image_encode_start

                prompts = batch["prompts"]
                maybe_sync_cuda(timing_sync_cuda)
                prompt_encode_start = time.perf_counter()
                prompt_embeds, prompt_attention_mask = encode_prompt(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt=prompts,
                    device=device,
                )
                maybe_sync_cuda(timing_sync_cuda)
                prompt_encode_time = time.perf_counter() - prompt_encode_start

            cpu_copy_time = 0.0
            save_submit_time = 0.0
            save_backpressure_time = 0.0
            save_completed_time = 0.0
            save_completed_count = 0

            maybe_sync_cuda(timing_sync_cuda)
            cpu_copy_start = time.perf_counter()
            if sync_save:
                vae_latents_cpu = vae_latents.detach().cpu()
                image_latents_cpu = image_latents.detach().cpu()
                fake_image_latents_cpu = fake_image_latents.detach().cpu()
                prompt_embeds_cpu = prompt_embeds.detach().cpu()
                copy_event = None
            else:
                vae_latents_cpu = async_copy_to_cpu(vae_latents.detach(), copy_stream)
                image_latents_cpu = async_copy_to_cpu(image_latents.detach(), copy_stream)
                fake_image_latents_cpu = async_copy_to_cpu(fake_image_latents.detach(), copy_stream)
                prompt_embeds_cpu = async_copy_to_cpu(prompt_embeds.detach(), copy_stream)
                copy_event = torch.cuda.Event()
                copy_event.record(copy_stream)
            first_frames_images_cpu = batch["first_frames_images"].to(torch.uint8).clone()
            cpu_copy_time = time.perf_counter() - cpu_copy_start

            batch_payloads = []
            for sample_idx, (uttid, num_frame, height, width, cur_prompt) in enumerate(
                zip(
                    batch["uttid"],
                    batch["video_metadata"]["num_frames"],
                    batch["video_metadata"]["height"],
                    batch["video_metadata"]["width"],
                    prompts,
                )
            ):
                batch_payloads.append(
                    {
                        "output_path": os.path.join(output_latent_folder, f"{uttid}_{num_frame}_{height}_{width}.pt"),
                        "vae_latent": vae_latents_cpu[sample_idx],
                        "image_latents": image_latents_cpu[sample_idx],
                        "fake_image_latents": fake_image_latents_cpu[sample_idx],
                        "prompt_embed": prompt_embeds_cpu[sample_idx],
                        "first_frames_image": first_frames_images_cpu[sample_idx],
                        "prompt_raw": cur_prompt,
                    }
                )

            if sync_save:
                sync_save_start = time.perf_counter()
                output_paths, completed_time, completed_count = save_batch_payloads(
                    batch_payloads,
                    copy_event,
                    (vae_latents, image_latents, fake_image_latents, prompt_embeds),
                )
                save_completed_time += completed_time
                save_completed_count += completed_count
                for output_path in output_paths:
                    print(f"save latent to: {output_path}")
                _ = time.perf_counter() - sync_save_start
            else:
                save_submit_start = time.perf_counter()
                save_futures.append(
                    save_executor.submit(
                        save_batch_payloads,
                        batch_payloads,
                        copy_event,
                        (vae_latents, image_latents, fake_image_latents, prompt_embeds),
                    )
                )
                save_submit_time += time.perf_counter() - save_submit_start

                save_futures, completed_time, completed_count = drain_save_futures(save_futures)
                save_completed_time += completed_time
                save_completed_count += completed_count

                if len(save_futures) >= max_pending_saves:
                    backpressure_start = time.perf_counter()
                    while len(save_futures) >= max_pending_saves:
                        save_futures, completed_time, completed_count = drain_save_futures(
                            save_futures, wait_for_one=True
                        )
                        save_completed_time += completed_time
                        save_completed_count += completed_count
                    save_backpressure_time += time.perf_counter() - backpressure_start

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({"batch": idx, "pending_saves": len(save_futures)})

            del pixel_values
            del prompts
            del vae_latents
            del image_latents
            del fake_image_latents
            del prompt_embeds
            del prompt_attention_mask
            del batch
            del valid_indices
            del valid_uttids
            del valid_num_frames
            del valid_heights
            del valid_widths
            del valid_videos
            del valid_prompts
            del valid_first_frames_images

            free_memory_after_time = 0.0
            if (idx + 1) % gc_interval == 0:
                free_memory_start = time.perf_counter()
                free_memory()
                free_memory_after_time = time.perf_counter() - free_memory_start

            if debug_timing and idx % timing_every == 0:
                log_timing(
                    global_rank,
                    idx,
                    {
                        "wait_batch_s": f"{wait_batch_time:.4f}",
                        "filter_s": f"{filter_time:.4f}",
                        "rebuild_s": f"{rebuild_time:.4f}",
                        "h2d_s": f"{h2d_time:.4f}",
                        "vae_s": f"{vae_encode_time:.4f}",
                        "image_s": f"{image_encode_time:.4f}",
                        "prompt_s": f"{prompt_encode_time:.4f}",
                        "cpu_copy_s": f"{cpu_copy_time:.4f}",
                        "save_submit_s": f"{save_submit_time:.4f}",
                        "save_backpressure_s": f"{save_backpressure_time:.4f}",
                        "save_completed_s": f"{save_completed_time:.4f}",
                        "free_mem_pre_s": f"{free_memory_before_time:.4f}",
                        "free_mem_mid_s": f"{free_memory_mid_time:.4f}",
                        "free_mem_post_s": f"{free_memory_after_time:.4f}",
                    },
                    {
                        "raw_batch": raw_batch_size,
                        "valid_batch": valid_batch_size,
                        "num_frames": num_frames,
                        "num_chunks": num_chunk_to_encode,
                        "pending_saves": len(save_futures),
                        "completed_saves": save_completed_count,
                    },
                )
            idx += 1
    finally:
        if not sync_save:
            while save_futures:
                save_futures, _, _ = drain_save_futures(save_futures, wait_for_one=True)
            save_executor.shutdown(wait=True)
        if rank == 0:
            pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running model training and data processing.")
    parser.add_argument("--dataloader_num_workers", type=int, default=16, help="Number of workers for data loading")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=8, help="Prefetch factor for each dataloader worker")
    parser.add_argument("--disable_persistent_workers", action="store_true", help="Disable persistent dataloader workers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per process")
    parser.add_argument("--debug_timing", action="store_true", help="Print per-rank timing breakdowns")
    parser.add_argument("--timing_every", type=int, default=10, help="Print timing every N steps when debugging")
    parser.add_argument(
        "--timing_sync_cuda",
        action="store_true",
        help="Synchronize CUDA around timed sections for more accurate GPU timing",
    )
    parser.add_argument("--save_workers", type=int, default=2, help="Number of background threads used for torch.save")
    parser.add_argument(
        "--max_pending_saves",
        type=int,
        default=128,
        help="Maximum number of outstanding async save jobs before applying backpressure",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/beijing-c/models/BestWishYSH/Helios-Base",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--base_video_path",
        type=str,
        default="/beijing-c/datasets/hxj_video_model/SpatialVID/SpatialVID",
        help="Base folder for source videos",
    )
    parser.add_argument(
        "--base_csv_path",
        type=str,
        default="/beijing-c/datasets/hxj_video_model/SpatialVID/SpatialVID",
        help="Base folder for input json metadata",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="helios_data_straight_5hz.part_002_of_002.json",
        help="Relative json path under base_csv_path",
    )
    parser.add_argument(
        "--base_output_latent_path",
        type=str,
        default="/world_model_data/SpatialVID/5fps",
        help="Base folder for latent output",
    )
    parser.add_argument(
        "--output_latent_path",
        type=str,
        default="latents_short_straight",
        help="Relative latent output folder under base_output_latent_path",
    )
    parser.add_argument("--resolution", type=int, default=640, help="Target resolution bucket")
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride")
    parser.add_argument(
        "--skip_video_scan",
        action="store_true",
        help="Bypass full os.walk over the video root and validate only candidate subset paths",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Only process a small subset of json records in debug mode",
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=42,
        help="Random seed used when --subset_mode=random",
    )
    parser.add_argument(
        "--subset_mode",
        type=str,
        choices=["head", "random"],
        default="head",
        help="How to choose the debug subset from the input json",
    )
    parser.add_argument(
        "--sync_save",
        action="store_true",
        help="Save latents synchronously on the main thread to mimic the regular export script",
    )
    args = parser.parse_args()

    setup_distributed_env()

    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()
    base_video_path = args.base_video_path
    video_paths = [
        "",
    ]
    base_output_latent_path = args.base_output_latent_path
    output_latent_paths = [
        args.output_latent_path,
    ]

    base_csv_paths = [
        args.base_csv_path,
    ]
    csv_paths = [
        args.csv_path,
    ]

    resolutions = [args.resolution]
    strides = [args.stride]
    batch_sizes = [args.batch_size]

    for stride, batch_size, base_csv_path, csv_path, video_path, output_latent_path, cur_resolution in zip(
        strides, batch_sizes, base_csv_paths, csv_paths, video_paths, output_latent_paths, resolutions
    ):
        json_file = os.path.join(base_csv_path, csv_path)
        video_folder = os.path.join(base_video_path, video_path)
        output_latent_folder = os.path.join(base_output_latent_path, output_latent_path)

        main(
            rank=device,
            world_size=world_size,
            global_rank=global_rank,
            stride=stride,
            batch_size=batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            dataloader_persistent_workers=not args.disable_persistent_workers,
            debug_timing=args.debug_timing,
            timing_every=args.timing_every,
            timing_sync_cuda=args.timing_sync_cuda,
            save_workers=args.save_workers,
            max_pending_saves=args.max_pending_saves,
            json_file=json_file,
            video_folder=video_folder,
            output_latent_folder=output_latent_folder,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            resolution=cur_resolution,
            skip_video_scan=args.skip_video_scan,
            subset_size=args.subset_size,
            subset_seed=args.subset_seed,
            subset_mode=args.subset_mode,
            sync_save=args.sync_save,
        )

    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
