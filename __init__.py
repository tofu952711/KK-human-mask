from comfy_api.latest import ComfyExtension
from .nodes import NODE_LIST


class VideoDepthExtension(ComfyExtension):
    async def get_node_list(self):
        return NODE_LIST


async def comfy_entrypoint():
    return VideoDepthExtension()
