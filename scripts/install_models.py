"""Explicit offline copy or user-triggered download. Never called during node import."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import urllib.request

MODELS = [
    ('sapiens2/sapiens2_0.4b_seg.safetensors', 'sapiens2-seg-0.4b/sapiens2_0.4b_seg.safetensors',
     'b85fdb50b7d6123a967d5ee4a505e222baff8d2f7ad6bbf353578c1a61dfbac9',
     'https://huggingface.co/facebook/sapiens2-seg-0.4b/resolve/main/sapiens2_0.4b_seg.safetensors'),
    ('video_depth_anything/video_depth_anything_vits.pth', 'video_depth_anything_vits.pth',
     '13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609',
     'https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth'),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfy-root', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', type=Path, help='Existing VideoDepth/models directory for offline installation')
    group.add_argument('--download', action='store_true', help='Download the two named weights explicitly')
    args = parser.parse_args()
    for relative, local, expected, url in MODELS:
        target = args.comfy_root.resolve()/'models'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != expected:
                raise RuntimeError(f'Existing file SHA256 mismatch; will not overwrite: {target}')
            print('Verified existing:', target, flush=True)
            continue
        staging = target.with_suffix(target.suffix+'.partial')
        if staging.exists():
            raise FileExistsError(f'Review/remove incomplete file before retry: {staging}')
        if args.source:
            source = args.source.resolve()/local
            if sha256(source) != expected:
                raise RuntimeError(f'Offline source SHA256 mismatch: {source}')
            try:
                os.link(source, staging)
            except OSError:
                shutil.copy2(source, staging)
        else:
            with urllib.request.urlopen(url, timeout=60) as response, staging.open('xb') as handle:
                shutil.copyfileobj(response, handle, 8*1024*1024)
        if sha256(staging) != expected:
            raise RuntimeError(f'Download/copy SHA256 mismatch: {staging}')
        if target.exists():
            raise FileExistsError(target)
        staging.rename(target)
        print('Ready:', target, flush=True)


if __name__ == '__main__':
    main()
