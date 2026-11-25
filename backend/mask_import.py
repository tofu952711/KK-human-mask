"""Reuse a matching exported mask without loading the Sapiens2 model."""
from pathlib import Path
import threading
import time

import cv2
import imageio_ffmpeg
import numpy as np

from .engine import check_cancel, encoder_process, target_size, video_info


def import_person_masks(source, mask_source, work, *, max_res=0, limit_frames=0, emit=None, cancel=None):
    emit = emit or (lambda *a, **k: None)
    cancel = cancel or threading.Event()
    source, mask_source = Path(source).resolve(), Path(mask_source).resolve()
    if source == mask_source:
        raise ValueError('请选黑白人体遮罩，不要选择原始彩色视频。')
    if not mask_source.is_file():
        raise ValueError('已有遮罩文件不存在。')
    info, mask_info = video_info(source), video_info(mask_source)
    if info['frames'] != mask_info['frames']:
        raise ValueError(f'已有遮罩与原视频帧数不一致：{mask_info["frames"]}/{info["frames"]}。请选择同一原片的遮罩。')
    if abs(info['fps'] - mask_info['fps']) > .01:
        raise ValueError('已有遮罩与原视频帧率不一致。')
    if abs(mask_info['width'] / mask_info['height'] - info['width'] / info['height']) > .005:
        raise ValueError('已有遮罩与原视频画面比例不一致。')
    check_cancel(cancel)
    width, height = target_size(info['width'], info['height'], max_res)
    expected = min(info['frames'], limit_frames) if limit_frames else info['frames']
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    packed, rendered, errors = work/'person-masks.bits', work/'person-mask.mp4', work/'mask-ffmpeg.log'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{width}x{height}', '-r', str(info['fps']),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '0',
               '-pix_fmt', 'yuvj420p', '-color_range', 'pc', '-movflags', '+faststart', str(rendered)]
    cap = cv2.VideoCapture(str(mask_source))
    count, empty = 0, 0
    started = time.monotonic()
    emit('log', text=f'复用已有遮罩，跳过 Sapiens2 识别：{mask_source.name}')
    try:
        with packed.open('wb') as bits, errors.open('wb') as err, encoder_process(command, err) as encoder:
            while count < expected:
                check_cancel(cancel)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f'已有遮罩解码提前结束：{count}/{expected} 帧。')
                if max(cv2.absdiff(bgr[:, :, 0], bgr[:, :, 1]).max(),
                       cv2.absdiff(bgr[:, :, 1], bgr[:, :, 2]).max()) > 8:
                    raise ValueError('所选视频包含彩色内容，请选择白人黑底的人体遮罩视频。')
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                mask = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST) >= 128
                np.packbits(mask.reshape(-1)).tofile(bits)
                gray = mask.astype(np.uint8)*255
                empty += not mask.any()
                encoder.stdin.write(gray.tobytes())
                count += 1
                if count % 24 == 0 or count == 1:
                    emit('progress', value=count/expected*99, text=f'读取已有遮罩 {count}/{expected} 帧')
            encoder.stdin.close()
            while encoder.poll() is None:
                check_cancel(cancel)
                time.sleep(.1)
            if encoder.returncode:
                raise RuntimeError(errors.read_text(encoding='utf-8', errors='replace'))
        if video_info(rendered)['frames'] != count:
            raise RuntimeError('复用遮罩的导出帧数不匹配。')
        return dict(packed_mask=str(packed), mask_video=str(rendered), frames=count, width=width,
                    height=height, fps=info['fps'], empty_frames=int(empty), seconds=round(time.monotonic()-started, 2),
                    model='已有人体遮罩', model_input=[mask_info['width'], mask_info['height']], precision='binary')
    finally:
        cap.release()
