"""Bounded-memory Video Depth Anything inference and atomic video export."""
from __future__ import annotations

import contextlib
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from .depth_effects import apply_depth_blur, blur_sigma, validate_blur

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
from .vendor.video_depth_anything.video_depth import VideoDepthAnything
from .vendor.video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from .vendor.utils.util import compute_scale_and_shift

MODEL_NAME = "video_depth_anything_vits.pth"
MODEL_URL = "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/" + MODEL_NAME
MODEL_SHA256 = "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609"
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()
    if cancel.is_set():
        raise Cancelled("任务已取消，未完成的视频已清理。")


def model_path():
    from ..model_paths import depth_path
    return depth_path()


def ensure_model(emit, cancel):
    path = model_path()
    emit("status", text="校验 Video Depth Anything 模型…")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            check_cancel(cancel)
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise RuntimeError(f"Depth model SHA256 mismatch: {path}")
    return path


def _probe_video_stream(probe, path, *, count_frames=False):
    command = [probe, '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames:stream_tags=rotate:stream_side_data=rotation',
               '-of', 'json']
    if count_frames:
        command.append('-count_frames')
    command += ['-i', str(Path(path).resolve())]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'ffprobe 无法读取视频信息。')
    streams = json.loads(result.stdout).get('streams', [])
    if not streams:
        raise ValueError('视频中没有可读取的视频轨道。')
    return streams[0]


def _video_info_ffprobe(probe, path):
    stream = _probe_video_stream(probe, path)
    fps = 0.0
    for key in ('avg_frame_rate', 'r_frame_rate'):
        try:
            fps = float(Fraction(stream.get(key, '0/0')))
        except (ValueError, ZeroDivisionError):
            continue
        if math.isfinite(fps) and fps > 0:
            break
    width, height = int(stream.get('width', 0)), int(stream.get('height', 0))
    rotation = stream.get('tags', {}).get('rotate', 0)
    for side_data in stream.get('side_data_list', []):
        if 'rotation' in side_data:
            rotation = side_data['rotation']
    if round(float(rotation)) % 180 == 90:
        width, height = height, width
    if width < 2 or height < 2 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('视频尺寸或帧率无效。')
    frames = str(stream.get('nb_frames', ''))
    if not frames.isdecimal() or int(frames) <= 0:
        counted = _probe_video_stream(probe, path, count_frames=True)
        frames = str(counted.get('nb_read_frames', ''))
    if not frames.isdecimal() or int(frames) <= 0:
        raise ValueError('视频中没有可解码的画面。')
    return {'width': width, 'height': height, 'fps': fps, 'frames': int(frames)}


def video_info(path):
    probe = os.environ.get('KK_MASK_FFPROBE') or shutil.which('ffprobe')
    if not probe:
        raise RuntimeError('缺少 ffprobe。请在云端安装 FFmpeg 工具包，或用 KK_MASK_FFPROBE 指定 ffprobe 路径。')
    return _video_info_ffprobe(probe, path)


def target_size(w, h, maximum):
    scale = min(1.0, maximum / max(w, h)) if maximum else 1.0
    # H.264 yuv420p requires even dimensions. Never crop the source.
    return max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2)


def normalized_gray(depth, low, high, invert=False):
    gray = np.round(np.clip((depth.astype(np.float32) - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)
    return 255 - gray if invert else gray


@contextlib.contextmanager
def encoder_process(cmd, error_stream):
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_stream,
                               creationflags=CREATE_NO_WINDOW)
    try:
        yield process
    finally:
        if process.poll() is None:
            with contextlib.suppress(Exception):
                process.kill()
                process.wait(timeout=5)
        with contextlib.suppress(Exception):
            process.stdin.close()


def extract(input_path, output_path, *, input_size=280, device="cuda", max_res=1280,
            invert=False, limit_frames=0, emit=None, cancel=None, mask_info=None, blur_amount=0):
    emit = emit or (lambda *a, **k: None)
    cancel = cancel or threading.Event()
    blur_amount = validate_blur(blur_amount)
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if not source.is_file():
        raise ValueError("输入视频不存在。")
    if source == destination:
        raise ValueError("输出不能覆盖原视频。")
    if destination.exists():
        raise FileExistsError("输出文件已存在，请更换文件名。")
    if device not in ("cuda", "cpu"):
        raise ValueError("请选择显卡模式或 CPU 模式。")
    if input_size not in (196, 280, 392, 518):
        raise ValueError("不支持的处理精度。")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 NVIDIA CUDA 显卡。请检查驱动，或手动选择 CPU 模式。")
    info = video_info(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_cancel(cancel)
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    emit("log", text=f"设备：{gpu_name}；模型：Video Depth Anything Small；精度：{input_size}")
    emit("log", text=f"原视频：{info['width']} × {info['height']}，{info['fps']:.6g} FPS，约 {info['frames']} 帧")
    weight = ensure_model(emit, cancel)
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
    emit("status", text="正在加载模型…")
    model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    output_w, output_h = target_size(info["width"], info["height"], max_res)
    if blur_amount:
        emit('log', text=f'成品深度模糊：{blur_amount}/100，最后对整张深度图柔化，包含人物轮廓。')
    mask_data = None
    if mask_info:
        if (mask_info['width'], mask_info['height']) != (output_w, output_h):
            raise ValueError('人物遮罩和深度输出尺寸不一致。')
        mask_data = np.memmap(mask_info['packed_mask'], dtype=np.uint8, mode='r',
                              shape=(mask_info['frames'], (output_w*output_h+7)//8))

    def person_mask(index, size=None):
        mask = np.unpackbits(mask_data[index], count=output_w*output_h).reshape(output_h, output_w)
        if size is not None:
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)
    # Match the official preprocessing, including wide-aspect memory control.
    ratio = max(info["width"], info["height"]) / min(info["width"], info["height"])
    size = input_size if ratio <= 1.78 else max(14, round(input_size * 1.777 / ratio / 14) * 14)
    transforms = [Resize(width=size, height=size, resize_target=False, keep_aspect_ratio=True,
                         ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
                  NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), PrepareForNet()]
    cap = cv2.VideoCapture(str(source))
    decoded = 0
    expected = min(info["frames"], limit_frames) if limit_frames else info["frames"]
    started = time.monotonic()
    ffmpeg = None
    try:
        with tempfile.TemporaryDirectory(prefix=".depth-work-", dir=destination.parent) as folder:
            work = Path(folder)
            spool = work / "depth.f32"
            rendered = work / "depth.mp4"
            error_log = work / "ffmpeg.log"
            written, low, high = 0, float("inf"), float("-inf")
            last_rgb = None
            shape = None

            def read_frames(count):
                nonlocal decoded, last_rgb
                values = []
                for _ in range(count):
                    check_cancel(cancel)
                    if limit_frames and decoded >= limit_frames:
                        break
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    decoded += 1
                    # Cap decoding working size, while retaining the output aspect ratio.
                    if max(bgr.shape[:2]) > 1280:
                        wh = target_size(bgr.shape[1], bgr.shape[0], 1280)
                        bgr = cv2.resize(bgr, wh, interpolation=cv2.INTER_AREA)
                    last_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    sample = {"image": last_rgb.astype(np.float32) / 255.0}
                    for transform in transforms:
                        sample = transform(sample)
                    values.append(torch.from_numpy(sample["image"]))
                return values

            with spool.open("wb") as raw:
                def append(depths):
                    nonlocal written, low, high
                    for depth in depths:
                        if written >= decoded:
                            return
                        check_cancel(cancel)
                        depth = np.maximum(depth, 0).astype(np.float32, copy=False)
                        if not np.isfinite(depth).all():
                            raise RuntimeError("模型输出异常，请降低处理精度后重试。")
                        depth.tofile(raw)
                        selected = depth[person_mask(written,(depth.shape[1],depth.shape[0]))] if mask_info else depth
                        if selected.size:
                            low, high = min(low, float(selected.min())), max(high, float(selected.max()))
                        written += 1

                frames = read_frames(32)
                if not frames:
                    raise ValueError("视频中没有可解码的画面。")
                previous_input, pending, references = None, None, None
                while frames:
                    check_cancel(cancel)
                    initial = previous_input is None
                    needed = 32 if initial else 22
                    frames += [frames[-1]] * (needed - len(frames))
                    current = torch.stack(frames).unsqueeze(0).to(device)
                    if not initial:
                        current = torch.cat((previous_input[:, KEYFRAMES], current), dim=1)
                    emit("status", text=f"显卡正在提取深度：已读取 {decoded} / {expected or '?'} 帧…" if device == "cuda" else f"CPU 正在提取深度：已读取 {decoded} 帧…")
                    with torch.autocast(device_type=device, enabled=device == "cuda"):
                        prediction = model(current)[0].float().cpu().numpy()
                    check_cancel(cancel)
                    shape = prediction.shape[1:]
                    if initial:
                        references = [prediction[0].copy(), prediction[12].copy()]
                        append(prediction[:24])
                        pending = prediction[24:].copy()
                    else:
                        a, b = compute_scale_and_shift(prediction[:2].reshape(-1), np.stack(references).reshape(-1),
                                                       np.ones(prediction[:2].size, dtype=bool))
                        if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
                            a, b = 1.0, 0.0
                        prediction = np.maximum(prediction * a + b, 0)
                        blend = np.linspace(0, 1, 8, dtype=np.float32)[:, None, None]
                        append(pending * (1 - blend) + prediction[2:10] * blend)
                        append(prediction[10:24])
                        pending = prediction[24:].copy()
                        references = [references[0], prediction[12].copy()]
                    previous_input = current
                    emit("progress", value=min(89, decoded / max(expected, decoded) * 89),
                         text=f"深度提取 {decoded}/{expected or '?'} 帧 · 已用 {time.monotonic()-started:.0f} 秒")
                    gray = normalized_gray(prediction[-1], low, max(high, low + 1e-6), invert)
                    if mask_info:
                        gray = np.where(person_mask(decoded-1,(gray.shape[1],gray.shape[0])),gray,0).astype(np.uint8)
                    gray = apply_depth_blur(gray, blur_amount)
                    emit("preview", original=last_rgb.copy(), depth=gray)
                    if limit_frames and decoded >= limit_frames:
                        break
                    frames = read_frames(22)
                append(pending)
            cap.release()
            del model, previous_input, current
            if device == "cuda":
                torch.cuda.empty_cache()
            if written != decoded:
                raise RuntimeError(f"帧数校验失败：读取 {decoded}，深度 {written}。")
            if mask_info and written != mask_info['frames']:
                raise RuntimeError('人物遮罩和深度视频帧数不一致。')
            if not math.isfinite(low) or not math.isfinite(high):
                low, high = 0.0, 1.0
            check_cancel(cancel)
            emit("log", text=f"深度提取完成，共 {written} 帧。使用全片统一亮度范围，避免逐帧拉伸造成闪烁。")
            emit("status", text="正在导出灰度深度视频…")
            encode_w, encode_h = (output_w,output_h) if mask_info or blur_amount else (shape[1],shape[0])
            cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{encode_w}x{encode_h}",
                   "-r", str(info["fps"]), "-i", "pipe:0", "-an", "-vf", f"scale={output_w}:{output_h}:flags=bilinear",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "0" if mask_info else "16",
                   "-pix_fmt", "yuvj420p" if mask_info else "yuv420p"]
            if mask_info:
                cmd += ['-color_range','pc']
            cmd += ["-movflags", "+faststart", str(rendered)]
            with error_log.open("wb") as err, spool.open("rb") as raw, encoder_process(cmd, err) as ffmpeg:
                for index in range(written):
                    check_cancel(cancel)
                    data = np.fromfile(raw, dtype=np.float32, count=shape[0] * shape[1]).reshape(shape)
                    gray = normalized_gray(data, low, high, invert)
                    if mask_info:
                        gray = cv2.resize(gray,(output_w,output_h),interpolation=cv2.INTER_LINEAR)
                        gray[~person_mask(index)] = 0
                    elif blur_amount:
                        gray = cv2.resize(gray,(output_w,output_h),interpolation=cv2.INTER_LINEAR)
                    gray = apply_depth_blur(gray, blur_amount)
                    try:
                        ffmpeg.stdin.write(gray.tobytes())
                    except BrokenPipeError:
                        raise RuntimeError("视频编码失败，请查看日志。") from None
                    if index % 20 == 0:
                        emit("progress", value=90 + 9 * (index + 1) / written, text=f"导出视频 {index+1}/{written} 帧")
                ffmpeg.stdin.close()
                while ffmpeg.poll() is None:
                    check_cancel(cancel)
                    time.sleep(0.1)
                if ffmpeg.returncode:
                    raise RuntimeError(error_log.read_text(encoding="utf-8", errors="replace"))
            check_cancel(cancel)
            verify = video_info(rendered)
            if verify["frames"] != written:
                raise RuntimeError("导出视频帧数不匹配，已停止保存。")
            if destination.exists():
                raise FileExistsError("输出文件已被其他程序创建，请更换文件名。")
            # Windows rename fails if a destination appeared concurrently, so never overwrite.
            rendered.rename(destination)
            result = {"output": str(destination), "frames": written, "fps": info["fps"], "width": output_w,
                      "height": output_h, "seconds": round(time.monotonic()-started, 2), "device": gpu_name,
                      "model": "Video Depth Anything Small", "input_size": input_size,
                      "normalization": "global_min_max", "near_white": not invert, "audio": False}
            result.update(blur_amount=blur_amount, blur_effect='fast_gaussian' if blur_amount else 'none',
                          blur_sigma_pixels=blur_sigma(blur_amount, output_w, output_h),
                          blur_stage='after_mask_at_export_resolution')
            metadata = destination.with_suffix(".json")
            if not metadata.exists():
                with contextlib.suppress(OSError):
                    metadata.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            emit("progress", value=100, text="提取完成")
            emit("done", **result)
            return result
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("显存不足。请选择「省显存 280」或「极速 196」，关闭其他占用显卡的软件后重试。") from None
    finally:
        cap.release()
        if mask_data is not None:
            mask_data._mmap.close()
        if ffmpeg is not None and ffmpeg.poll() is None:
            with contextlib.suppress(Exception):
                ffmpeg.kill()
                ffmpeg.wait(timeout=5)
        if device == "cuda":
            torch.cuda.empty_cache()
