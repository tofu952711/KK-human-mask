"""Generate importable ComfyUI UI workflows and API prompts without extra extensions."""
import json
import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parents[1]/'examples'
root.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('example_models', root.parent/'scripts/install_models.py')
model_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_info)
models = [dict(name=Path(relative).name, url=url, directory=str(Path(relative).parent),
               hash=checksum, hash_type='SHA256')
          for relative, _, checksum, url in model_info.MODELS]
api = {
    '1': {'class_type':'NellyVD_LoadVideo','inputs':{'video':'your_video.mp4'}},
    '2': {'class_type':'NellyVD_PersonVideo','inputs':{'video':['1',0], 'device':'cuda',
          'depth_precision':'280','max_resolution':0,'invert':False,'blur':'关闭'}},
    '3': {'class_type':'NellyVD_SaveVideo','inputs':{'video':['2',0],'filename_prefix':'VideoDepth/person_depth'}},
    '4': {'class_type':'NellyVD_SaveVideo','inputs':{'video':['2',1],'filename_prefix':'VideoDepth/person_mask'}},
}
(root/'person_depth_api.json').write_text(json.dumps(api,ensure_ascii=False,indent=2),encoding='utf-8')
def node(i, typ, pos, inputs, outputs, widgets, title):
    return dict(id=i,type=typ,pos=pos,size=[300,220],flags={},order=i-1,mode=0,
                inputs=[dict(name=name,type=kind,link=link) for name,kind,link in inputs],
                outputs=[dict(name=name,type=kind,links=links,slot_index=j) for j,(name,kind,links) in enumerate(outputs)],
                properties={'Node name for S&R':typ},widgets_values=widgets,title=title)
graph = dict(last_node_id=4,last_link_id=3,nodes=[
    node(1,'NellyVD_LoadVideo',[40,100],[],[('VIDEO','VIDEO',[1])],['your_video.mp4'],'1 选择彩色原片'),
    node(2,'NellyVD_PersonVideo',[400,100],[('video','VIDEO',1)],
         [('DEPTH_VIDEO','VIDEO',[2]),('MASK_VIDEO','VIDEO',[3]),('SHARP_DEPTH_VIDEO','VIDEO',[]),('INFO','STRING',[])],
         ['cuda','280',0,False,'关闭'],'KK人体遮罩'),
    node(3,'NellyVD_SaveVideo',[820,100],[('video','VIDEO',2)],
         [('FILE_PATH','STRING',[])],['VideoDepth/person_depth'],'3 保存深度'),
    node(4,'NellyVD_SaveVideo',[820,430],[('video','VIDEO',3)],
         [('FILE_PATH','STRING',[])],['VideoDepth/person_mask'],'4 保存清晰人体遮罩'),
],links=[[1,1,0,2,0,'VIDEO'],[2,2,0,3,0,'VIDEO'],[3,2,1,4,0,'VIDEO']],
groups=[],config={},extra={'ds':{'scale':.8,'offset':[20,40]}},version=.4,models=models)
(root/'person_depth.json').write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
blur_api={
    '1':{'class_type':'NellyVD_LoadVideo','inputs':{'video':'your_depth_video.mp4'}},
    '2':{'class_type':'NellyVD_FinalBlur','inputs':{'depth_video':['1',0],'blur':'开启','blur.amount':30}},
    '3':{'class_type':'NellyVD_SaveVideo','inputs':{'video':['2',0],'filename_prefix':'VideoDepth/blurred'}},
}
(root/'blur_only_api.json').write_text(json.dumps(blur_api,ensure_ascii=False,indent=2),encoding='utf-8')
blur_graph=dict(last_node_id=3,last_link_id=2,nodes=[
    node(1,'NellyVD_LoadVideo',[40,100],[],[('VIDEO','VIDEO',[1])],['your_depth_video.mp4'],'选择已有灰度深度视频'),
    node(2,'NellyVD_FinalBlur',[400,100],[('depth_video','VIDEO',1)],[('DEPTH_VIDEO','VIDEO',[2])],['开启',30],'成品模糊'),
    node(3,'NellyVD_SaveVideo',[760,100],[('video','VIDEO',2)],[('FILE_PATH','STRING',[])],['VideoDepth/blurred'],'保存模糊深度'),
],links=[[1,1,0,2,0,'VIDEO'],[2,2,0,3,0,'VIDEO']],groups=[],config={},extra={},version=.4)
(root/'blur_only.json').write_text(json.dumps(blur_graph,ensure_ascii=False,indent=2),encoding='utf-8')
print('Example workflows generated.')
