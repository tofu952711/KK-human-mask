"""File-backed video nodes: cache masks/depth separately from the final blur."""
import gc
import json
import logging
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from comfy_api.latest import InputImpl, io, ui
from comfy_execution.graph_utils import GraphBuilder

from .backend import engine, sapiens_engine, mask_import
from .backend.depth_effects import apply_depth_blur, blur_depth_video

CATEGORY = 'KK人体遮罩'
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}


def work_directory():
    parent = Path(folder_paths.get_temp_directory())
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='videodepth-', dir=parent))


def video_path(video):
    source = video.get_stream_source()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f'Video file is missing: {path}. Requeue its producing node.')
        return path
    path = work_directory() / 'input.mp4'
    video.save_to(str(path))
    return path


def processing_device(device):
    if device == 'cpu':
        return 'cpu'
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select CPU explicitly or install CUDA PyTorch.')
        mm.unload_all_models()
        mm.soft_empty_cache()
        return 'cuda'
    raise ValueError('Device must be cuda or cpu')


def progress_callback():
    progress = ProgressBar(100)
    def emit(kind, **data):
        mm.throw_exception_if_processing_interrupted()
        if kind == 'progress':
            progress.update_absolute(min(100, int(data['value'])), 100)
        elif kind in ('log', 'status'):
            logging.info('[VideoDepth] %s', data.get('text', ''))
    return emit


class VDLoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        base = Path(folder_paths.get_input_directory())
        files = sorted(p.relative_to(base).as_posix() for p in base.rglob('*')
                       if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        return io.Schema(node_id='NellyVD_LoadVideo', display_name='KK 加载视频', category=CATEGORY,
                         inputs=[io.Combo.Input('video', options=files, upload=io.UploadType.video)],
                         outputs=[io.Video.Output('VIDEO')])

    @classmethod
    def execute(cls, video):
        path = Path(folder_paths.get_annotated_filepath(video)).resolve()
        base = Path(folder_paths.get_input_directory()).resolve()
        if not path.is_relative_to(base) or not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError('Select a video inside ComfyUI/input.')
        return io.NodeOutput(InputImpl.VideoFromFile(str(path)))

    @classmethod
    def fingerprint_inputs(cls, video):
        path = Path(folder_paths.get_annotated_filepath(video))
        return (path.stat().st_size, path.stat().st_mtime_ns)


class VDPersonMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='NellyVD_PersonMask', display_name='KK Sapiens2 人体遮罩（旧版）',
                         category='_VideoDepth/internal', is_deprecated=True,
                         description='保留可见人体、衣服与手部，尽量排除手持物；逐帧处理，不补画遮挡区域。',
                         inputs=[io.Video.Input('video'), io.Combo.Input('device', options=['cuda', 'cpu']),
                                 io.Int.Input('max_resolution', default=0, min=0, max=8192, step=2,
                                              tooltip='输出最长边；0=原尺寸。模型固定宽768×高1024。')],
                         outputs=[io.Video.Output('MASK_VIDEO'), io.String.Output('INFO')])

    @classmethod
    def execute(cls, video, device, max_resolution):
        source = video_path(video)
        work = work_directory()
        try:
            result = sapiens_engine.create_person_masks(source, work, device=processing_device(device),
                                                        max_res=max_resolution, emit=progress_callback())
            path = result['mask_video']
            Path(result['packed_mask']).unlink()
            return io.NodeOutput(InputImpl.VideoFromFile(path), json.dumps(result, ensure_ascii=False))
        except BaseException:
            shutil.rmtree(work)
            raise


class VDPersonDepth(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='NellyVD_PersonDepth', display_name='KK 视频人体深度（旧版）',
                         category='_VideoDepth/internal', is_deprecated=True,
                         description='从彩色原片计算时序深度，按可选遮罩裁切。不要把抠像后黑底画面当原片。',
                         inputs=[io.Video.Input('video'), io.Combo.Input('device', options=['cuda', 'cpu']),
                                 io.Combo.Input('depth_precision', options=['280', '196', '392', '518']),
                                 io.Int.Input('max_resolution', default=0, min=0, max=8192, step=2),
                                 io.Boolean.Input('invert', default=False),
                                 io.Video.Input('mask_video', optional=True)],
                         outputs=[io.Video.Output('DEPTH_VIDEO'), io.String.Output('INFO')])

    @classmethod
    def execute(cls, video, device, depth_precision, max_resolution, invert, mask_video=None):
        source = video_path(video)
        work = work_directory()
        masks = None
        emit = progress_callback()
        try:
            if mask_video is not None:
                masks = mask_import.import_person_masks(source, video_path(mask_video), work,
                                                        max_res=max_resolution, emit=emit)
            result = engine.extract(source, work/'depth.mp4', input_size=int(depth_precision),
                                    device=processing_device(device), max_res=max_resolution,
                                    invert=invert, mask_info=masks, emit=emit)
            if masks is not None:
                Path(masks['packed_mask']).unlink()
                Path(masks['mask_video']).unlink()
            return io.NodeOutput(InputImpl.VideoFromFile(result['output']), json.dumps(result, ensure_ascii=False))
        except BaseException:
            shutil.rmtree(work)
            raise
        finally:
            gc.collect()
            mm.soft_empty_cache()


def blur_option():
    return io.DynamicCombo.Input('blur', options=[
        io.DynamicCombo.Option('关闭', []),
        io.DynamicCombo.Option('开启', [io.Int.Input('amount', default=30, min=0, max=100,
                                                   display_mode=io.NumberDisplay.slider)]),
    ])


class VDPersonVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='NellyVD_PersonVideo', display_name='KK人体遮罩',
            category=CATEGORY, enable_expand=True,
            description='接入彩色原片，自动识别人体并提取人体深度；可直接开启成品模糊。只改模糊时复用内部 AI 缓存。',
            inputs=[io.Video.Input('video', raw_link=True),
                    io.Combo.Input('device', options=['cuda', 'cpu']),
                    io.Combo.Input('depth_precision', options=['280', '196', '392', '518']),
                    io.Int.Input('max_resolution', default=0, min=0, max=8192, step=2,
                                 tooltip='输出最长边；0=原尺寸。'),
                    io.Boolean.Input('invert', default=False), blur_option()],
            outputs=[io.Video.Output('DEPTH_VIDEO'), io.Video.Output('MASK_VIDEO'),
                     io.Video.Output('SHARP_DEPTH_VIDEO'), io.String.Output('INFO')])

    @classmethod
    def execute(cls, video, device, depth_precision, max_resolution, invert, blur):
        # Deterministic internal nodes let ComfyUI keep AI results when only blur changes.
        graph = GraphBuilder()
        mask = graph.node('NellyVD_PersonMask', id='mask', video=video,
                          device=device, max_resolution=max_resolution)
        depth = graph.node('NellyVD_PersonDepth', id='depth', video=video,
                           mask_video=mask.out(0), device=device, depth_precision=depth_precision,
                           max_resolution=max_resolution, invert=invert)
        blur_inputs = {'depth_video': depth.out(0), 'blur': blur['blur']}
        if blur['blur'] == '开启':
            blur_inputs['blur.amount'] = blur['amount']
        final = graph.node('NellyVD_FinalBlur', id='blur', **blur_inputs)
        return io.NodeOutput(final.out(0), mask.out(0), depth.out(0), depth.out(1), expand=graph.finalize())


class VDFinalBlur(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='NellyVD_FinalBlur', display_name='KK 成品深度模糊', category=CATEGORY,
                         description='开启后显示0–100拉条；只处理已生成深度，不重跑AI。关闭直接复用输入。',
                         inputs=[io.Video.Input('depth_video'), blur_option()],
                         outputs=[io.Video.Output('DEPTH_VIDEO')])

    @classmethod
    def execute(cls, depth_video, blur):
        if blur['blur'] == '关闭':
            return io.NodeOutput(depth_video)
        if blur['blur'] != '开启':
            raise ValueError('Invalid blur switch')
        work = work_directory()
        try:
            result = blur_depth_video(video_path(depth_video), work/'blur.mp4',
                                      blur_amount=blur['amount'], emit=progress_callback())
            return io.NodeOutput(InputImpl.VideoFromFile(result['output']))
        except BaseException:
            shutil.rmtree(work)
            raise


class VDSaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='NellyVD_SaveVideo', display_name='KK 保存深度或遮罩视频', category=CATEGORY,
                         is_output_node=True, inputs=[io.Video.Input('video'),
                         io.String.Input('filename_prefix', default='VideoDepth/depth')],
                         outputs=[io.String.Output('FILE_PATH')])

    @classmethod
    def execute(cls, video, filename_prefix):
        source = video_path(video)
        out = Path(folder_paths.get_output_directory()).resolve()
        prefix = Path(filename_prefix)
        if prefix.is_absolute() or '..' in prefix.parts or '\\' in filename_prefix or not filename_prefix.strip():
            raise ValueError('Use a relative output prefix, e.g. VideoDepth/depth.')
        directory, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, str(out))
        directory = Path(directory).resolve()
        if not directory.is_relative_to(out):
            raise ValueError('Output must stay inside ComfyUI/output.')
        directory.mkdir(parents=True, exist_ok=True)
        while True:
            target = directory/f'{name}_{counter:05d}.mp4'
            try:
                with target.open('xb') as handle, source.open('rb') as inp:
                    shutil.copyfileobj(inp, handle)
                break
            except FileExistsError:
                counter += 1
            except BaseException:
                target.unlink(missing_ok=True)
                raise
        saved = ui.SavedResult(target.name, subfolder, io.FolderType.output)
        return io.NodeOutput(str(target), ui=ui.PreviewVideo([saved]))


class VDBlurFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='NellyVD_BlurFrames', display_name='KK 深度帧模糊（IMAGE）', category=CATEGORY,
                         inputs=[io.Image.Input('images'), blur_option()], outputs=[io.Image.Output('IMAGE')])

    @classmethod
    def execute(cls, images, blur):
        if blur['blur'] == '关闭':
            return io.NodeOutput(images)
        if blur['blur'] != '开启':
            raise ValueError('Invalid blur switch')
        result = torch.empty_like(images, device='cpu')
        progress = ProgressBar(len(images))
        for i, image in enumerate(images):
            mm.throw_exception_if_processing_interrupted()
            frame = np.round(image.detach().cpu().numpy().clip(0, 1)*255).astype(np.uint8)
            result[i] = torch.from_numpy(apply_depth_blur(frame, blur['amount']).copy()).float()/255
            progress.update(1)
        return io.NodeOutput(result)


NODE_LIST = [VDLoadVideo, VDPersonVideo, VDPersonMask, VDPersonDepth, VDFinalBlur, VDSaveVideo, VDBlurFrames]
