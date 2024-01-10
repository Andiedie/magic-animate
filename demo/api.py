import urllib.request
from PIL import Image
from diffusers.utils import load_image
from io import BytesIO
import tempfile
from demo.animate import MagicAnimate
import numpy as np
from pathlib import Path
from flask import Flask, request
import base64

app = Flask(__name__)

animator = MagicAnimate()

global_dict = {
    'progress': 0
}


@app.get('/ping')
def ping():
    return 'pong'


@app.post('/invoke')
def invoke():
    global_dict['progress'] = 0

    data = request.get_json()
    img_data = data['image']
    pose_data = data['pose']
    seed = data['seed']
    steps = data['steps']
    guidance_scale = data['guidance_scale']
    size = data['size']

    if img_data.startswith('https://'):
        image = load_image(img_data)
    else:
        img_bytes = base64.b64decode(img_data.encode())
        image = Image.open(BytesIO(img_bytes))
        image = load_image(image)

    image = np.array(image)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        pose_file_path = temp_file.name
        urllib.request.urlretrieve(pose_data, pose_file_path)

    def progress_callback(stage: str, current_step: int, _total_steps: int):
        if stage == 'denoise':
            global_dict['progress'] = (current_step * 20) / (steps * 20 + 48 + 1)
        if stage == 'decode':
            global_dict['progress'] = (steps * 20 + current_step) / (steps * 20 + 48 + 1)

    result_path = animator(image, pose_file_path, seed, steps, guidance_scale, size, progress_callback)

    result_path = Path.cwd().joinpath(result_path).resolve().absolute()
    video_b64 = base64.b64encode(result_path.read_bytes()).decode()
    result_path.unlink()

    return {'video': video_b64}


@app.get('/progress')
def progress():
    return {
        'progress': global_dict['progress']
    }
