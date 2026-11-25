# Third-party materials

- Sapiens2 0.4B: Meta Platforms, Inc. and affiliates. https://github.com/facebookresearch/sapiens2 . Sapiens2 License: `licenses/Sapiens2-LICENSE.md`. Standalone backbone retains copyright notice. Head layout adapted from https://github.com/starsFriday/ComfyUI-Sapiens2 . Weight SHA256: `b85fdb50b7d6123a967d5ee4a505e222baff8d2f7ad6bbf353578c1a61dfbac9`.
- Video Depth Anything Small: Bytedance Ltd. and/or affiliates. https://github.com/DepthAnything/Video-Depth-Anything . Upstream revision `4f5ae23172ba60fd7bc11ef671cca678842c7072`. `licenses/Video-Depth-Anything-LICENSE.txt`. Small weight SHA256: `13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609`.
- DINOv2 and other files inside vendored Video Depth Anything retain original Meta / upstream headers and associated upstream license files.
- Local adaptations: relative package imports, explicit local model discovery, PyTorch SDPA attention implementation, disk-spooled depth export, ComfyUI progress and interruption, and scaled Gaussian blur. No global `sys.path` injection or global monkey-patching of third-party model classes.
- Runtime dependencies: PyTorch/torchvision, NumPy, OpenCV, imageio-ffmpeg (bundled Linux FFmpeg), safetensors, easydict, einops, Pillow and ComfyUI. Their own licenses apply. FFmpeg build information: https://github.com/imageio/imageio-binaries .

The Sapiens2 license permits royalty-free uses subject to its terms and restricts surveillance, biometric processing, deepfakes and impersonation, among other uses. Read the full license before deployment or redistribution.
