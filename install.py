"""ComfyUI Manager installation hook: verify or fetch the two model assets."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys


def find_comfy_root(explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    elif os.environ.get('COMFYUI_PATH'):
        candidates = [Path(os.environ['COMFYUI_PATH'])]
    else:
        candidates = [Path.cwd(), *Path(__file__).absolute().parents, *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / 'folder_paths.py').is_file() and (candidate / 'models').is_dir():
            return candidate.resolve()
    raise RuntimeError('Cannot locate ComfyUI. Run install.py --comfy-root /path/to/ComfyUI')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path)
    args = parser.parse_args()
    root = find_comfy_root(args.comfy_root)
    installer = Path(__file__).resolve().parent / 'scripts' / 'install_models.py'
    spec = importlib.util.spec_from_file_location('kk_mask_model_installer', installer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(installer), '--comfy-root', str(root), '--download']
    module.main()


if __name__ == '__main__':
    main()
