"""Run with the Linux ComfyUI Python: python tests/test_linux.py --comfy-root PATH."""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument('--comfy-root', type=Path, required=True)
args, rest = parser.parse_known_args()
sys.argv = [sys.argv[0], *rest]
sys.path.insert(0, str(args.comfy_root.resolve()))
import torch
import cv2
import numpy as np
import folder_paths

plugin_path = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('videodepth_test_plugin', plugin_path/'__init__.py',
                                            submodule_search_locations=[str(plugin_path)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
from videodepth_test_plugin import nodes
from videodepth_test_plugin.backend import engine
from videodepth_test_plugin.backend.depth_effects import apply_depth_blur, validate_blur


class FakeDepth:
    def __init__(self, **kwargs): pass
    def load_state_dict(self, *a, **k): pass
    def to(self, *a, **k): return self
    def eval(self): return self
    def __call__(self, x):
        base = torch.linspace(1, 2, 28).reshape(1, 1, 1, 28).expand(1, 32, 14, 28)
        return base + x[:, :, 0].mean(dim=(-2, -1))[:, :, None, None]*.1


class LinuxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vd-test-', dir=folder_paths.get_temp_directory())
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, n):
        path = self.root/f'中文 video {n}.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 24, (56, 28))
        for i in range(n): writer.write(np.full((28, 56, 3), 40+i, np.uint8))
        writer.release()
        return path

    def fake_extract(self, source, target, **kwargs):
        with patch.object(engine, 'VideoDepthAnything', FakeDepth), \
             patch.object(engine, 'ensure_model', return_value=Path('unused')), \
             patch.object(engine.torch, 'load', return_value={}):
            return engine.extract(source, target, device='cpu', input_size=196, max_res=0, **kwargs)

    def test_registration_and_schemas(self):
        extension = asyncio.run(plugin.comfy_entrypoint())
        listed = asyncio.run(extension.get_node_list())
        self.assertEqual(len(listed), 7)
        ids = [node.define_schema().node_id for node in listed]
        self.assertEqual(len(ids), len(set(ids)))

    def test_ffprobe_reads_mp4_and_mkv(self):
        source = self.source(5)
        self.assertEqual(engine.video_info(source), dict(width=56, height=28, fps=24.0, frames=5))
        target = self.root / '中文 video.mkv'
        subprocess.run([engine.imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
                        '-i', str(source), '-c', 'copy', str(target)],
                       stdin=subprocess.DEVNULL, capture_output=True, check=True)
        self.assertEqual(engine.video_info(target), dict(width=56, height=28, fps=24.0, frames=5))
        rotated = self.root / 'rotated.mp4'
        subprocess.run([engine.imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
                        '-display_rotation', '90', '-i', str(source), '-c', 'copy', str(rotated)],
                       stdin=subprocess.DEVNULL, capture_output=True, check=True)
        self.assertEqual(engine.video_info(rotated), dict(width=28, height=56, fps=24.0, frames=5))
        cap = cv2.VideoCapture(str(rotated))
        try:
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (56, 28))
        finally:
            cap.release()

    def test_ffprobe_command_has_no_ffmpeg_flags(self):
        stream = dict(width=56, height=28, avg_frame_rate='30000/1001', nb_frames='5')
        response = subprocess.CompletedProcess([], 0, json.dumps({'streams': [stream]}), '')
        with patch.object(engine.subprocess, 'run', return_value=response) as run:
            info = engine._video_info_ffprobe('/usr/bin/ffprobe', self.root / '中文 video.mp4')
        self.assertAlmostEqual(info['fps'], 30000 / 1001)
        command = run.call_args.args[0]
        self.assertEqual(command[0], '/usr/bin/ffprobe')
        for option in ['-nostdin', '-y', '-c:v', '-pix_fmt']:
            self.assertNotIn(option, command)
        self.assertIs(run.call_args.kwargs['stdin'], subprocess.DEVNULL)

    def test_ffprobe_errors_and_missing_binary(self):
        bad = self.root / 'broken.mp4'
        bad.write_bytes(b'not a video')
        with self.assertRaises(RuntimeError):
            engine.video_info(bad)
        with patch.dict(os.environ, {}, clear=True), patch.object(engine.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, '缺少 ffprobe'):
                engine.video_info(bad)
        probe = shutil.which('ffprobe')
        with patch.dict(os.environ, {'KK_MASK_FFPROBE': probe}), \
             patch.object(engine.shutil, 'which', side_effect=AssertionError('explicit path ignored')):
            self.assertEqual(engine.video_info(self.source(1))['frames'], 1)

    def test_combined_node_keeps_ai_graph_stable(self):
        from comfy_execution.graph_utils import GraphBuilder
        graphs = []
        for blur in [{'blur':'关闭'}, {'blur':'开启','amount':30}, {'blur':'开启','amount':60}]:
            GraphBuilder.set_default_prefix('combined-test', 0)
            result = nodes.VDPersonVideo.execute(['1',0], 'cuda', '280', 0, False, blur)
            graphs.append(result.expand)
            self.assertEqual(len(result.result), 4)
            self.assertEqual(result.result[1], ['combined-test.0.0.mask', 0])
        for graph in graphs:
            self.assertEqual(graph['combined-test.0.0.mask']['inputs']['video'], ['1',0])
            self.assertEqual(graph['combined-test.0.0.depth']['inputs']['mask_video'], ['combined-test.0.0.mask',0])
        for key in ['combined-test.0.0.mask', 'combined-test.0.0.depth']:
            self.assertEqual(graphs[0][key], graphs[1][key])
            self.assertEqual(graphs[1][key], graphs[2][key])
        self.assertEqual(graphs[2]['combined-test.0.0.blur']['inputs']['blur.amount'], 60)

    def test_temporal_boundaries_and_mask_zero(self):
        for n in [1, 32, 33, 54, 55]:
            source = self.source(n)
            mask = np.zeros((28, 56), np.uint8);mask[4:24, 8:48] = 1;mask[10:16, 30:40] = 0
            bits = self.root/f'{n}.bits'
            np.tile(np.packbits(mask), (n, 1)).tofile(bits)
            info = dict(width=56, height=28, frames=n, packed_mask=str(bits))
            target = self.root/f'depth-{n}.mp4'
            result = self.fake_extract(source, target, mask_info=info)
            self.assertEqual(result['frames'], n)
            cap = cv2.VideoCapture(str(target))
            try:
                for i in range(n):
                    ok, frame = cap.read();self.assertTrue(ok)
                    self.assertEqual(int(frame[mask==0].max()), 0)
            finally: cap.release()

    def test_cancel_and_output_preservation(self):
        source = self.source(33);original = source.read_bytes();stop = threading.Event()
        def emit(kind, **data):
            if kind == 'progress': stop.set()
        target = self.root/'cancel.mp4'
        with self.assertRaises(engine.Cancelled): self.fake_extract(source, target, cancel=stop, emit=emit)
        self.assertFalse(target.exists())
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse(list(self.root.glob('.depth-work-*')))

    def test_blur_disabled_identity_and_enabled_spread(self):
        image = np.zeros((360, 270), np.uint8);image[100:260, 80:190] = 200
        self.assertIs(apply_depth_blur(image, 0), image)
        blurred = apply_depth_blur(image, 100)
        self.assertGreater(int(blurred[180, 70]), 0)
        self.assertLess(int(blurred[100, 80]), 200)
        for value in [-1,101,float('nan')]:
            with self.assertRaises(ValueError): validate_blur(value)
        tensor = torch.rand(2, 16, 20, 3)
        result = nodes.VDBlurFrames.execute(tensor, {'blur':'关闭'})
        self.assertIs(result.result[0], tensor)

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError): nodes.VDLoadVideo.execute('../escape.mp4')
        path = self.source(1)
        video = nodes.InputImpl.VideoFromFile(str(path))
        with self.assertRaises(ValueError): nodes.VDSaveVideo.execute(video, '../escape')

    def test_video_blur_does_not_load_models(self):
        source = self.source(4)
        video = nodes.InputImpl.VideoFromFile(str(source))
        with patch.object(engine, 'VideoDepthAnything', side_effect=AssertionError('no model')):
            result = nodes.VDFinalBlur.execute(video, {'blur':'开启','amount':30})
        self.assertEqual(engine.video_info(result.result[0].get_stream_source())['frames'], 4)


if __name__ == '__main__':
    Path(folder_paths.get_temp_directory()).mkdir(parents=True, exist_ok=True)
    unittest.main(verbosity=2)
