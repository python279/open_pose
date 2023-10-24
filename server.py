from open_pose import MyOpenposeDetector
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from base64 import b64decode, b64encode


app = Flask(__name__)

open_pose = MyOpenposeDetector.from_pretrained("./checkpoint/models--lllyasviel--Annotators/snapshots/982e7edaec38759d914a963c48c4726685de7d96")

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()  # 获取 POST 请求中的 JSON 数据
    input_image = data["input_image"]
    # 从 base64 解码出图片
    image = Image.open(BytesIO(b64decode(input_image))).convert("RGB")
    output_image, coco_keypoints = open_pose(image, detect_resolution=768, image_resolution=768,
                                             include_body=True, include_hand=True, include_face=True)
    buffered = BytesIO()
    output_image.save(buffered, format='PNG')
    output_image = b64encode(buffered.getvalue()).decode('utf-8')
    result = {'result': {'output_image': output_image, 'coco_keypoints': coco_keypoints}}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
