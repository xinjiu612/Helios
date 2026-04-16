import argparse
import os
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from accelerate import Accelerator
from helios.dataset.dataloader_mp4_dist import BucketedFeatureDataset, BucketedSampler, collate_fn
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


def metadata_cache_path(json_file):
    return json_file.replace(".json", "_cache.pkl").replace(".csv", "_cache.pkl")


def metadata_cache_ready_path(cache_file):
    return f"{cache_file}.ready"


def wait_for_metadata_cache(cache_file, ready_file, wait_started_at, timeout_s=7200, poll_interval_s=2):
    while True:
        if os.path.exists(cache_file) and os.path.exists(ready_file):
            try:
                if os.path.getmtime(ready_file) >= wait_started_at:
                    return
            except OSError:
                pass

        elapsed = time.time() - wait_started_at
        if elapsed > timeout_s:
            raise TimeoutError(f"Timed out waiting for metadata cache to be ready: {cache_file}")
        time.sleep(poll_interval_s)


def ensure_metadata_cache(
    json_file,
    video_folder,
    stride,
    resolution,
    force_rebuild,
    global_rank,
):
    cache_file = metadata_cache_path(json_file)
    ready_file = metadata_cache_ready_path(cache_file)
    wait_started_at = time.time()
    should_build = force_rebuild or not os.path.exists(cache_file)

    if should_build:
        if global_rank == 0:
            if os.path.exists(ready_file):
                os.remove(ready_file)

            print(f"Rank 0 preparing metadata cache: {cache_file}")
            BucketedFeatureDataset(
                json_files=json_file,
                video_folders=video_folder,
                stride=stride,
                force_rebuild=True,
                resolution=resolution,
                single_res=True,
                single_height=384,
                single_width=640,
            )
            with open(ready_file, "w") as f:
                f.write(f"{time.time()}\n")
        else:
            print(f"Rank {global_rank} waiting for metadata cache: {cache_file}")
            wait_for_metadata_cache(cache_file, ready_file, wait_started_at)

    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Metadata cache was not created successfully: {cache_file}")

    return cache_file


def async_copy_to_cpu(tensor, copy_stream):
    cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
    with torch.cuda.stream(copy_stream):
        cpu_tensor.copy_(tensor, non_blocking=True)
    return cpu_tensor


def save_batch_payloads(batch_payloads, copy_event=None, retained_gpu_tensors=None):
    if copy_event is not None:
        copy_event.synchronize()
    del retained_gpu_tensors

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
    return output_paths, len(batch_payloads)


def expected_latent_filename(sample_info):
    return (
        f"{sample_info['uttid']}_{sample_info['bucket_num_frame']}_"
        f"{sample_info['bucket_height']}_{sample_info['bucket_width']}.pt"
    )


def filter_completed_samples(dataset, output_latent_folder, global_rank):
    original_count = len(dataset.samples)
    if original_count == 0:
        return 0

    if not os.path.isdir(output_latent_folder):
        if global_rank == 0:
            print(f"Completed latent filter: output folder does not exist yet: {output_latent_folder}")
        return 0

    scan_start = time.time()
    completed_files = {
        entry.name
        for entry in os.scandir(output_latent_folder)
        if entry.is_file() and entry.name.endswith(".pt")
    }

    if not completed_files:
        if global_rank == 0:
            print(f"Completed latent filter: no existing .pt files in {output_latent_folder}")
        return 0

    filtered_samples = []
    filtered_buckets = defaultdict(list)
    skipped_count = 0

    for sample_info in dataset.samples:
        if expected_latent_filename(sample_info) in completed_files:
            skipped_count += 1
            continue

        new_idx = len(filtered_samples)
        filtered_samples.append(sample_info)
        filtered_buckets[sample_info["bucket_key"]].append(new_idx)

    dataset.samples = filtered_samples
    dataset.buckets = filtered_buckets

    if global_rank == 0:
        elapsed = time.time() - scan_start
        print(
            f"Completed latent filter: skipped {skipped_count}/{original_count} existing samples, "
            f"remaining {len(dataset.samples)} samples, scanned {len(completed_files)} files in {elapsed:.2f}s"
        )

    return skipped_count


def drain_save_futures(save_futures, wait_for_one=False):
    if not save_futures:
        return [], 0

    if wait_for_one:
        done, not_done = wait(save_futures, return_when=FIRST_COMPLETED)
        completed = list(done)
        remaining = list(not_done)
    else:
        completed = [future for future in save_futures if future.done()]
        remaining = [future for future in save_futures if not future.done()]

    for future in completed:
        try:
            output_paths, _ = future.result()
            for output_path in output_paths:
                print(f"save latent to: {output_path}")
        except Exception as exc:
            print(f"Save task failed: {exc}", flush=True)

    return remaining, len(completed)


def main(
    rank,
    world_size,
    global_rank,
    stride,
    batch_size,
    dataloader_num_workers,
    dataloader_prefetch_factor,
    dataloader_persistent_workers,
    save_workers,
    max_pending_saves,
    sync_save,
    force_rebuild,
    filter_completed,
    gc_interval,
    json_file,
    video_folder,
    output_latent_folder,
    pretrained_model_name_or_path,
    resolution=640,
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

    ensure_metadata_cache(
        json_file=json_file,
        video_folder=video_folder,
        stride=stride,
        resolution=resolution,
        force_rebuild=force_rebuild,
        global_rank=global_rank,
    )

    # Load the tokenizers
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

    # dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    dataset = BucketedFeatureDataset(
        json_files=json_file,
        video_folders=video_folder,
        stride=stride,
        force_rebuild=False,
        resolution=resolution,
        single_res=True,
        single_height=384,
        single_width=640,
    )
    if filter_completed:
        filter_completed_samples(dataset, output_latent_folder, global_rank)

    sampler = BucketedSampler(dataset, batch_size=batch_size, drop_last=True, shuffle=True, seed=seed)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        persistent_workers=dataloader_persistent_workers and dataloader_num_workers > 0,
        prefetch_factor=dataloader_prefetch_factor if dataloader_num_workers != 0 else None,
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
    try:
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            if batch is None or batch["videos"] is None:
                print("None batch, continuing")
                idx += 1
                continue

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

            if not valid_indices:
                print("skipping entire batch!")
                if rank == 0:
                    pbar.update(1)
                    pbar.set_postfix({"batch": idx, "pending_saves": len(save_futures)})
                idx += 1
                continue

            batch = None
            del batch

            batch = {
                "uttid": valid_uttids,
                "video_metadata": {"num_frames": valid_num_frames, "height": valid_heights, "width": valid_widths},
                "videos": torch.stack(valid_videos),
                "prompts": valid_prompts,
                "first_frames_images": torch.stack(valid_first_frames_images),
            }

            if len(batch["uttid"]) == 0:
                print("All samples in this batch are already processed, skipping!")
                idx += 1
                continue

            with torch.no_grad():
                pixel_values = batch["videos"].permute(0, 2, 1, 3, 4).to(dtype=vae.dtype, device=device, non_blocking=True)

                latent_window_size = 9
                frame_window_size = (latent_window_size - 1) * 4 + 1
                num_latent_frames = pixel_values.shape[2]
                num_chunk_to_encode = num_latent_frames // frame_window_size

                history_latent_list = []
                for i in range(num_chunk_to_encode):
                    start_idx = i * frame_window_size
                    end_idx = start_idx + frame_window_size
                    cur_pixel_values = pixel_values[:, :, start_idx:end_idx, :, :]
                    cur_latent = vae.encode(cur_pixel_values).latent_dist.sample()
                    cur_latent = (cur_latent - latents_mean) * latents_std
                    history_latent_list.append(cur_latent)
                vae_latents = torch.stack(history_latent_list, dim=1)

                image_pixel_values = pixel_values[:, :, :1, :, :]
                image_latents = vae.encode(image_pixel_values).latent_dist.sample()
                image_latents = (image_latents - latents_mean) * latents_std

                fake_video = image_pixel_values.repeat(1, 1, frame_window_size, 1, 1)
                fake_image_latents = vae.encode(fake_video).latent_dist.sample()
                fake_image_latents = (fake_image_latents - latents_mean) * latents_std
                fake_image_latents = fake_image_latents[:, :, -1:, :, :]

                prompts = batch["prompts"]
                prompt_embeds, prompt_attention_mask = encode_prompt(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt=prompts,
                    device=device,
                )

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
                output_paths, _ = save_batch_payloads(
                    batch_payloads,
                    copy_event,
                    (vae_latents, image_latents, fake_image_latents, prompt_embeds),
                )
                for output_path in output_paths:
                    print(f"save latent to: {output_path}")
            else:
                save_futures.append(
                    save_executor.submit(
                        save_batch_payloads,
                        batch_payloads,
                        copy_event,
                        (vae_latents, image_latents, fake_image_latents, prompt_embeds),
                    )
                )
                save_futures, _ = drain_save_futures(save_futures)

                if len(save_futures) >= max_pending_saves:
                    while len(save_futures) >= max_pending_saves:
                        save_futures, _ = drain_save_futures(save_futures, wait_for_one=True)

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

            if gc_interval > 0 and (idx + 1) % gc_interval == 0:
                free_memory()
            idx += 1
    finally:
        if not sync_save:
            while save_futures:
                save_futures, _ = drain_save_futures(save_futures, wait_for_one=True)
            save_executor.shutdown(wait=True)
        if rank == 0:
            pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running model training and data processing.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4, help="Prefetch factor for each dataloader worker")
    parser.add_argument("--disable_persistent_workers", action="store_true", help="Disable persistent dataloader workers")
    parser.add_argument("--save_workers", type=int, default=2, help="Number of background threads used for torch.save")
    parser.add_argument(
        "--max_pending_saves",
        type=int,
        default=128,
        help="Maximum number of outstanding async save jobs before applying backpressure",
    )
    parser.add_argument(
        "--sync_save",
        action="store_true",
        help="Save latents synchronously on the main thread instead of using async save workers",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Rebuild metadata cache on rank 0 before all ranks load it",
    )
    parser.add_argument(
        "--disable_completed_filter",
        action="store_true",
        help="Disable startup filtering of samples whose latent .pt already exists",
    )
    parser.add_argument(
        "--gc_interval",
        type=int,
        default=50,
        help="Call free_memory every N batches; set to 0 to disable periodic cleanup",
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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per process")
    args = parser.parse_args()

    setup_distributed_env()

    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()
    # `data.json` stores paths like `videos/InteriorGS/xxx.mp4`,
    # so video_folder must point to sampled_data root.
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
            save_workers=args.save_workers,
            max_pending_saves=args.max_pending_saves,
            sync_save=args.sync_save,
            force_rebuild=args.force_rebuild,
            filter_completed=not args.disable_completed_filter,
            gc_interval=args.gc_interval,
            json_file=json_file,
            video_folder=video_folder,
            output_latent_folder=output_latent_folder,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            resolution=cur_resolution,
        )

    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    dist.destroy_process_group()
