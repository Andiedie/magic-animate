import json
import sys
import traceback
import urllib.request
from PIL import Image
from diffusers.utils import load_image
import base64
from io import BytesIO
import tempfile
from demo.animate import MagicAnimate
import numpy as np

animator = MagicAnimate()

while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break

        data = json.loads(line)
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

        result_path = animator(image, pose_file_path, seed, steps, guidance_scale, size)

        print('data:' + json.dumps({'result': result_path}))
    except Exception as e:
        print('data:' + json.dumps({'err': traceback.format_exc()}))
