"""Resolve weights from native ComfyUI model folders, never download at import."""
from pathlib import Path
import folder_paths

SAPIENS_FILE = 'sapiens2_0.4b_seg.safetensors'
DEPTH_FILE = 'video_depth_anything_vits.pth'

folder_paths.add_model_folder_path('videodepth_sapiens2', str(Path(folder_paths.models_dir) / 'sapiens2'))
folder_paths.add_model_folder_path('videodepth_vda', str(Path(folder_paths.models_dir) / 'video_depth_anything'))


def sapiens_path():
    return Path(folder_paths.get_full_path_or_raise('videodepth_sapiens2', SAPIENS_FILE))


def depth_path():
    return Path(folder_paths.get_full_path_or_raise('videodepth_vda', DEPTH_FILE))
