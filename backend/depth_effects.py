"""Final depth-frame effects, applied after masking at export resolution."""
import math

import cv2


def validate_blur(amount):
    try:
        value = float(amount)
    except (ValueError, TypeError):
        raise ValueError('模糊幅度必须是 0–100 的整数。') from None
    if not math.isfinite(value) or not 0 <= value <= 100 or not value.is_integer():
        raise ValueError('模糊幅度必须是 0–100 的整数。')
    return int(value)


def blur_sigma(amount, width, height):
    # Resolution-relative radius keeps the visual strength consistent on export.
    return validate_blur(amount) / 100 * min(width, height) * .03


def apply_depth_blur(gray, amount):
    amount = validate_blur(amount)
    if not amount:
        return gray
    sigma = blur_sigma(amount, gray.shape[1], gray.shape[0])
    # Blur the finished depth, including its outline against the black background.
    # Do not apply the person mask again: that would restore a hard boundary.
    scale = min(8, max(1, int(sigma / 3)))
    if scale == 1:
        return cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REPLICATE)
    height, width = gray.shape[:2]
    small_w, small_h = max(1, round(width / scale)), max(1, round(height / scale))
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma * small_w / width,
                             sigmaY=sigma * small_h / height, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def blur_depth_video(source, destination, *, blur_amount=30, max_res=0, limit_frames=0,
                     emit=None, cancel=None):
    """Blur an already rendered depth video, without loading any neural network."""
    import contextlib
    import json
    from pathlib import Path
    import tempfile
    import threading
    import time
    import imageio_ffmpeg
    from .engine import check_cancel, encoder_process, target_size, video_info

    emit = emit or (lambda *a, **k: None)
    cancel = cancel or threading.Event()
    amount = validate_blur(blur_amount)
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise ValueError('请选择已生成的灰度深度视频。')
    if source == destination or destination.exists():
        raise ValueError('输出文件已存在或与输入相同，请更换文件名。')
    if limit_frames < 0:
        raise ValueError('处理帧数不能为负数。')
    check_cancel(cancel)
    info = video_info(source)
    width, height = target_size(info['width'], info['height'], max_res)
    expected = min(info['frames'], limit_frames) if limit_frames else info['frames']
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    count = 0
    cap = cv2.VideoCapture(str(source))
    cv2.setNumThreads(4)
    emit('log', text=f'已有深度加模糊：{amount}/100，不运行人体识别或深度模型。')
    try:
        with tempfile.TemporaryDirectory(prefix='.blur-work-', dir=destination.parent) as folder:
            work = Path(folder)
            rendered, errors = work/'blur.mp4', work/'ffmpeg.log'
            cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                   '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{width}x{height}', '-r', str(info['fps']),
                   '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '0',
                   '-pix_fmt', 'yuvj420p', '-color_range', 'pc', '-movflags', '+faststart', str(rendered)]
            with errors.open('wb') as err, encoder_process(cmd, err) as encoder:
                while count < expected:
                    check_cancel(cancel)
                    ok, bgr = cap.read()
                    if not ok:
                        raise RuntimeError(f'深度视频解码提前结束：{count}/{expected} 帧。')
                    if count == 0 and max(cv2.absdiff(bgr[:,:,0], bgr[:,:,1]).max(),
                                           cv2.absdiff(bgr[:,:,1], bgr[:,:,2]).max()) > 8:
                        raise ValueError('请选择已生成的灰度深度视频，不要选择彩色原片。')
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    if gray.shape != (height, width):
                        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
                    gray = apply_depth_blur(gray, amount)
                    check_cancel(cancel)
                    try:
                        encoder.stdin.write(gray.tobytes())
                    except BrokenPipeError:
                        raise RuntimeError('模糊深度视频编码失败。') from None
                    count += 1
                    if count == 1 or count % 24 == 0:
                        emit('progress', value=min(99, count/expected*99), text=f'深度模糊 {count}/{expected} 帧')
                        emit('preview', original=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth=gray)
                encoder.stdin.close()
                while encoder.poll() is None:
                    check_cancel(cancel)
                    time.sleep(.1)
                if encoder.returncode:
                    raise RuntimeError(errors.read_text(encoding='utf-8', errors='replace'))
            check_cancel(cancel)
            if video_info(rendered)['frames'] != count:
                raise RuntimeError('模糊深度视频导出帧数不一致。')
            if destination.exists():
                raise FileExistsError('输出文件刚被其他程序创建，请更换文件名。')
            rendered.rename(destination)
        result = dict(output=str(destination), frames=count, fps=info['fps'], width=width, height=height,
                      seconds=round(time.monotonic()-started,2), mode='blur', model='无需AI模型', audio=False,
                      blur_amount=amount, blur_effect='fast_gaussian' if amount else 'none',
                      blur_sigma_pixels=blur_sigma(amount,width,height), source=str(source), device='CPU')
        with contextlib.suppress(OSError):
            with destination.with_suffix('.json').open('x', encoding='utf-8') as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
        emit('progress', value=100, text='模糊完成')
        emit('done', **result)
        return result
    finally:
        cap.release()
