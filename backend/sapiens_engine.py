"""Offline Sapiens2 human parsing, matching the verified video-matting experiment."""
from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .engine import BASE, check_cancel, encoder_process, target_size, video_info

from .vendor.sapiens2_model import Sapiens2PersonSegmentation

MODEL_NAME = 'Sapiens2-seg-0.4B'
MODEL_FILE = 'sapiens2-seg-0.4b/sapiens2_0.4b_seg.safetensors'
MODEL_SHA256 = 'b85fdb50b7d6123a967d5ee4a505e222baff8d2f7ad6bbf353578c1a61dfbac9'
INPUT_SIZE = (1024, 768)  # height, width


def model_path():
    from ..model_paths import sapiens_path
    return sapiens_path()


def verify_model(cancel=None):
    path = model_path()
    if not path.is_file():
        raise RuntimeError(f'缺少 Sapiens2 人体模型，请恢复 models 文件夹：{path}')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            if cancel is not None:
                check_cancel(cancel)
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise RuntimeError(f'Sapiens2 人体模型校验失败，请恢复原模型文件：{path}')
    return path


def load_model(device, cancel=None):
    path = verify_model(cancel)
    dtype = torch.float16 if device == 'cuda' else torch.float32
    model = Sapiens2PersonSegmentation().eval().to(dtype=dtype)
    if cancel is not None:
        check_cancel(cancel)
    # Strict loading prevents partially initialized models from producing plausible masks.
    model.load_state_dict(load_file(str(path)), strict=True)
    if cancel is not None:
        check_cancel(cancel)
    return model.to(device)


def predict_person_mask(model, rgb, output_size, device):
    """Use every non-background body-part class; no thresholding or morphology."""
    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)
    image = F.interpolate(image, INPUT_SIZE, mode='bilinear', align_corners=False)
    mean = image.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    std = image.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    image = (image - mean) / std
    with torch.autocast(device_type=device, enabled=device == 'cuda', dtype=torch.float16):
        logits = model(image.to(dtype=torch.float16 if device == 'cuda' else torch.float32))
    logits = F.interpolate(logits.float(), size=(output_size[1], output_size[0]),
                           mode='bilinear', align_corners=False)
    if not torch.isfinite(logits).all():
        raise RuntimeError('Sapiens2 人体模型输出异常，请检查模型和显卡。')
    return logits.argmax(dim=1)[0].gt(0).cpu().numpy()


def create_person_masks(source, work, *, device='cuda', max_res=1280,
                        limit_frames=0, emit=None, cancel=None):
    emit = emit or (lambda *a, **k: None)
    cancel = cancel or threading.Event()
    if device not in ('cuda', 'cpu'):
        raise ValueError('请选择显卡或 CPU 模式。')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('未检测到 NVIDIA 显卡，人体遮罩不会自动切换到 CPU。')
    check_cancel(cancel)
    info = video_info(source)
    width, height = target_size(info['width'], info['height'], max_res)
    expected = min(info['frames'], limit_frames) if limit_frames else info['frames']
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    packed, rendered, errors = work / 'person-masks.bits', work / 'person-mask.mp4', work / 'mask-ffmpeg.log'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{width}x{height}', '-r', str(info['fps']),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '0',
               '-pix_fmt', 'yuvj420p', '-color_range', 'pc', '-movflags', '+faststart', str(rendered)]
    model = None
    cap = None
    count, empty = 0, 0
    started = time.monotonic()
    try:
        emit('status', text='正在校验并加载 Sapiens2 人体模型…')
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
        model = load_model(device, cancel)
        emit('log', text='人体遮罩：Sapiens2 0.4B · 保留头发、衣服、手和四肢，排除背景及手持物。')
        cap = cv2.VideoCapture(str(source))
        with packed.open('wb') as bits, errors.open('wb') as err, encoder_process(command, err) as encoder:
            while True:
                check_cancel(cancel)
                if limit_frames and count >= limit_frames:
                    break
                ok, bgr = cap.read()
                if not ok:
                    break
                # Run the model on original RGB, independent of the chosen export size.
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mask = predict_person_mask(model, rgb, (width, height), device)
                check_cancel(cancel)
                np.packbits(mask.reshape(-1)).tofile(bits)
                gray = mask.astype(np.uint8) * 255
                empty += not mask.any()
                try:
                    encoder.stdin.write(gray.tobytes())
                except BrokenPipeError:
                    raise RuntimeError('人体遮罩视频编码失败。') from None
                count += 1
                if count % 12 == 0 or count == 1:
                    emit('progress', value=min(99, count / max(expected, count) * 99),
                         text=f'人体遮罩 {count}/{expected or "?"} 帧 · 已用 {time.monotonic()-started:.0f} 秒')
                    emit('preview', original=rgb, depth=gray)
            if not count:
                raise ValueError('视频中没有可解码的画面。')
            encoder.stdin.close()
            while encoder.poll() is None:
                check_cancel(cancel)
                time.sleep(.1)
            if encoder.returncode:
                raise RuntimeError(errors.read_text(encoding='utf-8', errors='replace'))
        check_cancel(cancel)
        if expected > 0 and count != expected:
            raise RuntimeError(f'视频解码帧数不匹配：{count}/{expected} 帧。')
        if video_info(rendered)['frames'] != count:
            raise RuntimeError('人体遮罩导出帧数不匹配。')
        emit('log', text=f'Sapiens2 人体遮罩完成：{count} 帧，其中 {empty} 帧未检测到人物。')
        return dict(packed_mask=str(packed), mask_video=str(rendered), frames=count, width=width, height=height,
                    fps=info['fps'], empty_frames=int(empty), seconds=round(time.monotonic()-started, 2),
                    model=MODEL_NAME, model_input=[INPUT_SIZE[1], INPUT_SIZE[0]],
                    precision='fp16' if device == 'cuda' else 'fp32')
    finally:
        if cap is not None:
            cap.release()
        del model
        # Release the segmentation model before loading the depth model on a 6GB GPU.
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
