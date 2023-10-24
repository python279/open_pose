from controlnet_aux import OpenposeDetector
from controlnet_aux.util import HWC3, resize_image
from controlnet_aux.open_pose import draw_poses
import numpy as np
import cv2
from PIL import Image
import json


class MyOpenposeDetector(OpenposeDetector):
    """
    A class for detecting human poses in images using the Openpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """

    def __init__(self, body_estimation, hand_estimation=None, face_estimation=None):
        super().__init__(body_estimation, hand_estimation, face_estimation)

    def poses_to_coco_keypoints(self, poses, W, H):
        """
        Converts a pose to a coco keypoints array.

        Args:
            pose (dict): A pose dictionary.

        Returns:
            np.ndarray: A numpy array of shape (17, 3) containing the coco keypoints.
        """
        coco_keypoints = {
            "version": "1.3",
            "people": []
        }
        print(poses)
        for id, pose in enumerate(poses):
            coco_keypoints["people"].append({
                "pose_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "face_keypoints_2d": []
            })
            print(pose.body.keypoints)
            if pose.body.keypoints:
                for keypoint in pose.body.keypoints:
                    if keypoint is None:
                        coco_keypoints["people"][id]["pose_keypoints_2d"].append(0)
                        coco_keypoints["people"][id]["pose_keypoints_2d"].append(0)
                        coco_keypoints["people"][id]["pose_keypoints_2d"].append(0)
                    else:
                        x, y, score = keypoint.x, keypoint.y, keypoint.score
                        coco_keypoints["people"][id]["pose_keypoints_2d"].append(x * float(H))
                        coco_keypoints["people"][id]["pose_keypoints_2d"].append(y * float(W))
                        coco_keypoints["people"][id]["pose_keypoints_2d"].append(score)
            if pose.left_hand:
                for keypoint in pose.left_hand:
                    if keypoint is None:
                        coco_keypoints["people"][id]["hand_left_keypoints_2d"].append(0)
                        coco_keypoints["people"][id]["hand_left_keypoints_2d"].append(0)
                        coco_keypoints["people"][id]["hand_left_keypoints_2d"].append(0)
                    else:
                        x, y, score = keypoint.x, keypoint.y, keypoint.score
                        coco_keypoints["people"][id]["hand_left_keypoints_2d"].append(x * float(H))
                        coco_keypoints["people"][id]["hand_left_keypoints_2d"].append(y * float(W))
                        coco_keypoints["people"][id]["hand_left_keypoints_2d"].append(score)
            if pose.right_hand:
                for keypoint in pose.right_hand:
                    if keypoint is None:
                        coco_keypoints["people"][id]["hand_right_keypoints_2d"].append(0)
                        coco_keypoints["people"][id]["hand_right_keypoints_2d"].append(0)
                        coco_keypoints["people"][id]["hand_right_keypoints_2d"].append(0)
                    else:
                        x, y, score = keypoint.x, keypoint.y, keypoint.score
                        coco_keypoints["people"][id]["hand_right_keypoints_2d"].append(x * float(H))
                        coco_keypoints["people"][id]["hand_right_keypoints_2d"].append(y * float(W))
                        coco_keypoints["people"][id]["hand_right_keypoints_2d"].append(score)
            if pose.face:
                for keypoint in pose.face:
                    if keypoint is None:
                        coco_keypoints["people"][id]["face_keypoints_2d"].append(0)
                        coco_keypoints["people"][id]["face_keypoints_2d"].append(0)
                        coco_keypoints["people"][id]["face_keypoints_2d"].append(0)
                    else:
                        x, y, score = keypoint.x, keypoint.y, keypoint.score
                        coco_keypoints["people"][id]["face_keypoints_2d"].append(x * float(H))
                        coco_keypoints["people"][id]["face_keypoints_2d"].append(y * float(W))
                        coco_keypoints["people"][id]["face_keypoints_2d"].append(score)

        return coco_keypoints

    def __call__(self, input_image, detect_resolution=512, image_resolution=512, include_body=True, include_hand=False,
                 include_face=False, hand_and_face=True, output_type="pil", **kwargs):
        if hand_and_face is not None:
            include_hand = hand_and_face
            include_face = hand_and_face

        if "return_pil" in kwargs:
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            if output_type:
                output_type = "pil"

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape

        poses = self.detect_poses(input_image, include_hand, include_face)
        canvas = draw_poses(poses, H, W, draw_body=include_body, draw_hand=include_hand, draw_face=include_face)

        detected_map = canvas
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map, self.poses_to_coco_keypoints(poses, W, H)


if __name__ == '__main__':
    img = Image.open("00006_00.jpg").convert("RGB")
    open_pose = MyOpenposeDetector.from_pretrained("lllyasviel/Annotators")
    processed_image_open_pose, coco_keypoints = open_pose(img, detect_resolution=768, image_resolution=768, include_body=True, include_hand=True, include_face=True)
    processed_image_open_pose.save("00006_00_rendered.png")
    with open("00006_00_keypoints.json", "w") as f:
        json.dump(coco_keypoints, f)
