# KK人体遮罩

将本地 VideoDepth 软件的人体遮罩、时序人物深度与成品模糊移植到 ComfyUI。面向 Linux + NVIDIA CUDA 开发，也提供显式 CPU 选项。全部本地处理，不连接收费 API，不上传视频。

## 安装

使用现有 Linux ComfyUI 的 Python 环境。建议 Python 3.12、PyTorch 2.7+，需要支持 V3 `ComfyExtension`、VIDEO 和 DynamicCombo 的较新 ComfyUI。

GitHub 仓库和发布 ZIP 均不包含模型权重，只提供原始模型下载地址、目标目录和 SHA256。插件包包含完整代码、示例、依赖清单、安装脚本与上游许可证。

### 手动安装

从 [Releases](https://github.com/tofu952711/KK-human-mask/releases/latest) 下载 `KK-human-mask-1.1.1-linux.zip`，解压到 `ComfyUI/custom_nodes/KK人体遮罩`，或克隆仓库到该目录：

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/tofu952711/KK-human-mask.git KK人体遮罩
/path/to/comfy-venv/bin/python -m pip install -r KK人体遮罩/requirements.txt
/path/to/comfy-venv/bin/python KK人体遮罩/install.py --comfy-root /path/to/ComfyUI
```

不要从 Windows 软件包复制 `_internal`、`.exe`、DLL 或 Windows 的 FFmpeg 到 Linux。不安装或替换 PyTorch，不需要 Tkinter、ComfyUI-Manager、VideoHelperSuite、TensorRT 或编译 CUDA 扩展。

`requirements.txt` 按官方约定仅安装 Python 依赖，并用注释列出模型引用；`pyproject.toml` 的 dependencies 与它一致。`install.py` 是官方文档支持的 ComfyUI Manager 安装钩子：Manager 安装依赖后可执行它；手动安装则执行上面的命令。安装钩子校验已有模型，缺失时从原始地址下载，校验成功后放入标准模型目录。不要仅执行 pip 命令就认为模型也已安装。

安装沿用 ComfyUI 的 PyTorch/CUDA，不替换它们。模型需要约 1.63 GiB 空间，只需准备一次：

```text
ComfyUI/models/
  sapiens2/sapiens2_0.4b_seg.safetensors
  video_depth_anything/video_depth_anything_vits.pth
```

| 模型 | 原始下载地址 | 保存目录（相对 ComfyUI/models） |
| --- | --- | --- |
| Sapiens2 0.4B 人体分割 | [sapiens2_0.4b_seg.safetensors](https://huggingface.co/facebook/sapiens2-seg-0.4b/resolve/main/sapiens2_0.4b_seg.safetensors) | `sapiens2/` |
| Video Depth Anything Small | [video_depth_anything_vits.pth](https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth) | `video_depth_anything/` |

示例工作流 `examples/person_depth.json` 的顶层 `models` 字段使用官方 Workflow JSON 0.4 规范，记录两份模型的 `name`、`url`、`directory`、`hash` 和 `hash_type`。实际下载由安装脚本执行，不能把模型 URL 当作 requirements 中的 pip 安装项。

离线安装可从现有软件的 `models` 目录导入，校验 SHA256，同一文件系统优先硬链接以节省空间：

```bash
python custom_nodes/KK人体遮罩/scripts/install_models.py \
  --comfy-root /path/to/ComfyUI --source /path/to/VideoDepth/models
```

也可单独执行模型下载脚本：

```bash
python custom_nodes/KK人体遮罩/scripts/install_models.py \
  --comfy-root /path/to/ComfyUI --download
```

节点导入与执行不自动下载，不访问外网。支持 `extra_model_paths.yaml` 下的 `videodepth_sapiens2` 与 `videodepth_vda` 模型分类。

本插件目前通过 GitHub 发布，尚未登记为 Comfy Registry 的可搜索条目。安装文件依据官方文档：[Manager 安装约定](https://docs.comfy.org/custom-nodes/backend/manager)、[pyproject.toml 规范](https://docs.comfy.org/registry/specifications)、[Workflow JSON 0.4 模型引用](https://docs.comfy.org/specs/workflow_json_0.4)。

安装后重启 ComfyUI，搜索 `KK人体遮罩` 或进入“KK人体遮罩”。

已在 Linux / WSL2 Ubuntu 24.04、ComfyUI 0.45.0、PyTorch 2.7.1+cu118 上实际运行 CUDA 视频工作流。详细验证范围与耗时见 `VALIDATION.md`。

## 节点与连接

1. **KK 加载视频**：从 `ComfyUI/input` 选择/上传视频，输出原生 VIDEO。
2. **KK人体遮罩**：只接一次彩色原视频，自动生成遮罩、计算原视频时序深度，再将人体外置黑。包含成品模糊开关：默认“关闭”，切到“开启”才显示 0–100 幅度拉条，初始 30。输出 `DEPTH_VIDEO`（最终成品）、`MASK_VIDEO`（清晰遮罩）、`SHARP_DEPTH_VIDEO`（未模糊深度）和 `INFO`。遮罩尽量保留人体、衣服、头发和手部，排除手持物。
3. **KK 保存深度或遮罩视频**：保存无音轨 MP4 到 `ComfyUI/output`，提供原生视频预览和输出路径。输入前缀例如 `VideoDepth/depth`。
4. **KK 成品深度模糊**、**KK 深度帧模糊（IMAGE）**：供已有深度单独加模糊；一体节点已经包含模糊，无须额外连接。

推荐视频流程：

```text
加载视频 → 人体遮罩与深度（一体） ─ DEPTH_VIDEO → 保存深度
                                └ MASK_VIDEO  → 保存遮罩（可选）
```

原先拆分的两个节点仅保留为内部实现与旧工作流兼容，标记为旧版。新工作流直接搜索“KK人体遮罩”，无需手动连接遮罩和深度。原节点 ID 保持不变。

已有深度只加模糊：加载灰度深度视频 → 成品深度模糊 → 保存视频，不跑任何 AI。反复调幅度请始终用未模糊的原始深度。

`examples/person_depth.json` 可拖进画布；`examples/person_depth_api.json` 是 API 格式的同一流程。先选择已上传的原视频，再执行。

## 速度、显存和缓存

- Sapiens2 采用与此前实测相同的宽 768 × 高 1024 输入，CUDA FP16、CPU FP32，不训练或改权重。
- 深度精度默认 280，6GB 显卡建议 196/280；392/518 需要更多显存。降低此值不改变人体识别输入。
- 输出最长边 `0` 表示原尺寸。保持原帧序和标称帧率；可变帧率会转为标称固定帧率。奇数尺寸为 MP4 兼容调整到偶数。
- 模型每个节点执行后释放，执行前通知 ComfyUI 卸载其他模型以腾出显存。不会在节点全局缓存模型或自动回退到 CPU。
- VIDEO 工作流逐帧解码、遮罩按位缓存、深度采用 32 帧窗口和磁盘缓存，避免整段视频变成巨大的 IMAGE 张量。IMAGE 模糊节点本身仍受输入帧批次占用内存影响。
- 一体节点内部使用 ComfyUI 子图与独立缓存。ComfyUI 缓存保留时，仅改模糊不会重新执行人体遮罩和深度 AI；改变深度精度也可复用遮罩。重启、清缓存或修改上游会重算。要长期复用请将结果保存到 output 后重新加载。
- 开启模糊后轮廓灰度会向周边黑底柔和扩散；模糊关闭时遮罩外深度为 0。
- 推理设备使用当前 CUDA 设备；多显卡由启动 ComfyUI 时的 `--cuda-device` / `CUDA_VISIBLE_DEVICES` 选择。
- 支持 ComfyUI 取消，取消/失败清理当前节点未完成文件，完成的中间视频保留在 ComfyUI/temp。ComfyUI 关闭后可清理 temp 释放磁盘；清理后需重跑被删文件的生产节点。
- 该移植不承诺 Linux 自动提高识别速度。主要耗时仍在 Sapiens2，模型效果与原软件尽量一致。

## 效果与许可

手持物排除不是 100% 准确。指尖、嘴边、字幕、发丝和运动模糊仍可能误分；看不见的身体部分不补画。深度是相对深度，不是实际距离。

适配代码采用 Apache-2.0；Sapiens2 权重与代码遵循自定义 Sapiens2 License，其他上游代码保留各自许可。详见 `THIRD_PARTY_NOTICES.md` 和 `licenses/`。打包不意味着获得额外用途授权。
