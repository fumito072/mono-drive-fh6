"""Expose the official Qwen-Drive VLM through its unchanged Qwen3.5 backbone.

Uses hard links for the original weights and tokenizer, and the same image/video
normalisation. No tensors are modified and no remote Python code is executed.
"""
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'models/qwen-drive-1.0-4b'
target=ROOT/'models/qwen-drive-1.0-4b-vlm'
target.mkdir(exist_ok=True)
config=json.loads((source/'config.json').read_text(encoding='utf-8'))
(target/'config.json').write_text(json.dumps(config['vlm_config'],indent=2),encoding='utf-8')
for name in ('model.safetensors','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json',
             'chat_template.jinja','generation_config.json','preprocessor_config.json'):
    dst=target/name
    if not dst.exists():
        os.link(source/name,dst)
video=json.loads((source/'preprocessor_config.json').read_text(encoding='utf-8'))
video.pop('image_processor_type',None)
video['video_processor_type']='Qwen3VLVideoProcessor'
video['size']={'shortest_edge':4096,'longest_edge':25165824}
(target/'video_preprocessor_config.json').write_text(json.dumps(video,indent=2),encoding='utf-8')
(target/'source.json').write_text(json.dumps(dict(model='Qwen/Qwen-Drive-1.0-4B',
    revision='28484089a7cc8c335cf5089fb0745cf7c49b6eaa',unchanged_weights=True,
    key_mapping={'^vlm\\.':''}),indent=2),encoding='utf-8')
print(target)
