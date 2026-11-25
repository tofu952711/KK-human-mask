"""Create a plugin release with model references; never include model weights."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import stat
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_NAME = 'KK人体遮罩'


def source_files():
    root = ROOT
    files = [root/name for name in ('.gitignore','__init__.py','install.py','nodes.py','model_paths.py','requirements.txt',
             'pyproject.toml','README.md','VALIDATION.md','LICENSE','THIRD_PARTY_NOTICES.md')]
    for directory in ('backend','licenses'):
        files += [path for path in (root/directory).rglob('*') if path.is_file()
                  and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo')]
    files += [root/name for name in ('examples/person_depth.json','examples/person_depth_api.json',
              'examples/blur_only.json','examples/blur_only_api.json',
              'scripts/install_models.py','scripts/create_examples.py',
              'scripts/package_release.py','tests/test_linux.py','tests/integration_combined.py',
              'tests/validation.json','tests/combined-validation.json')]
    return sorted(files)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    staging = output.with_suffix(output.suffix+'.partial')
    if output.exists() or staging.exists():
        raise FileExistsError('Release destination must be new: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    files = source_files()
    prefix = PLUGIN_NAME + '/'
    payload = [(path, prefix+path.relative_to(ROOT).as_posix()) for path in files]
    manifest = {'plugin':PLUGIN_NAME, 'full_models_included':False, 'files':{}}
    for path, name in payload:
        manifest['files'][name] = {'bytes':path.stat().st_size, 'sha256':sha256(path)}
    with zipfile.ZipFile(staging, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(files):
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for path, name in payload:
            archive.write(path, name)
        archive.writestr(prefix+'KK-human-mask-manifest.json',
                         json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(staging) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        forbidden = ('__pycache__','tests/runtime/','launch_validation_server','.mp4','.exe','linux_result_preview.jpg')
        assert not any(any(token in name for token in forbidden) for name in names)
        assert not any(name.endswith(('.pth','.safetensors','.onnx','.pt')) for name in names)
        for name, meta in manifest['files'].items():
            entry = archive.getinfo(name)
            assert not stat.S_ISLNK(entry.external_attr >> 16), 'Archive must contain real files'
            digest = hashlib.sha256()
            with archive.open(name) as handle:
                for block in iter(lambda:handle.read(8*1024*1024), b''):
                    digest.update(block)
            assert entry.file_size == meta['bytes'] and digest.hexdigest() == meta['sha256'], name
        for relative in ('examples/person_depth.json','examples/blur_only.json'):
            graph = json.loads(archive.read(prefix+relative))
            nodes = {node['id']:node for node in graph['nodes']}
            for link_id, src, src_slot, dst, dst_slot, kind in graph['links']:
                assert link_id in nodes[src]['outputs'][src_slot]['links']
                assert nodes[dst]['inputs'][dst_slot]['link'] == link_id
                assert nodes[src]['outputs'][src_slot]['type'] == nodes[dst]['inputs'][dst_slot]['type'] == kind
        combined = json.loads(archive.read(prefix+'examples/person_depth_api.json'))
        assert combined['2']['class_type'] == 'NellyVD_PersonVideo'
        assert not any(node['class_type'] in ('NellyVD_PersonMask','NellyVD_PersonDepth') for node in combined.values())
    if staging.stat().st_size >= 2*1024**3:
        raise RuntimeError('Asset exceeds GitHub per-file release size limit')
    staging.rename(output)
    digest = sha256(output)
    output.with_suffix(output.suffix+'.sha256').write_text(digest+'  '+output.name+'\n')
    print(json.dumps({'zip':str(output),'files':len(payload),'bytes':output.stat().st_size,
                      'models_included':False,'sha256':digest},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
