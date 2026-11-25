"""Validate the combined node and real expansion-cache reuse on an idle test server."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import uuid

import aiohttp


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='http://127.0.0.1:8197')
    parser.add_argument('--video', required=True, help='A short video already in ComfyUI/input')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    graph = {
        '1': {'class_type':'NellyVD_LoadVideo', 'inputs':{'video':args.video}},
        '2': {'class_type':'NellyVD_PersonVideo', 'inputs':{
            'video':['1',0], 'device':'cuda', 'depth_precision':'280',
            'max_resolution':540, 'invert':False, 'blur':'开启', 'blur.amount':30}},
        '3': {'class_type':'NellyVD_SaveVideo', 'inputs':{
            'video':['2',0], 'filename_prefix':'validation_combined/depth30'}},
        '4': {'class_type':'NellyVD_SaveVideo', 'inputs':{
            'video':['2',1], 'filename_prefix':'validation_combined/mask'}},
        '5': {'class_type':'NellyVD_SaveVideo', 'inputs':{
            'video':['2',2], 'filename_prefix':'validation_combined/sharp'}},
    }
    client = 'combined-validation-' + uuid.uuid4().hex
    report = []
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(args.server+'/ws?clientId='+client) as ws:
            for amount in (30, 60, 0):
                graph['2']['inputs']['blur'] = '关闭' if amount == 0 else '开启'
                if amount:
                    graph['2']['inputs']['blur.amount'] = amount
                else:
                    graph['2']['inputs'].pop('blur.amount', None)
                graph['3']['inputs']['filename_prefix'] = f'validation_combined/depth{amount}'
                started = time.monotonic()
                async with session.post(args.server+'/prompt', json={'prompt':graph,'client_id':client}) as response:
                    submitted = await response.json()
                    assert response.status == 200, submitted
                prompt_id = submitted['prompt_id']
                executed = []
                while True:
                    message = await ws.receive(timeout=300)
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(message.data)
                    data = event.get('data', {})
                    if data.get('prompt_id') != prompt_id:
                        continue
                    if event['type'] == 'executing' and data.get('node'):
                        executed.append(data['node'])
                    if event['type'] in ('execution_error','execution_interrupted'):
                        raise RuntimeError(data)
                    if event['type'] == 'execution_success':
                        break
                async with session.get(args.server+'/history/'+prompt_id) as response:
                    history = (await response.json())[prompt_id]
                assert history['status']['status_str'] == 'success'
                record = {'blur':amount, 'seconds':round(time.monotonic()-started,3),
                          'executed_nodes':executed, 'success':True}
                report.append(record)
                print(json.dumps(record), flush=True)
                ai = [node for node in executed if node.endswith(('.mask','.depth'))]
                if amount != 30:
                    assert not ai, f'Unexpected AI execution: {ai}'
                hashes = {}
                for node_id in ('4','5'):
                    saved = history['outputs'][node_id]['images'][0]
                    async with session.get(args.server+'/view', params={
                            key:saved[key] for key in ('filename','subfolder','type')}) as response:
                        assert response.status == 200
                        hashes[node_id] = hashlib.sha256(await response.read()).hexdigest()
                if amount == 30:
                    first_hashes = hashes
                else:
                    # Output filenames can change; underlying mask/sharp bytes must be identical.
                    assert hashes == first_hashes
    args.report.write_text(json.dumps({'combined_cuda_runs':report,'ai_cache_reused':True}, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
