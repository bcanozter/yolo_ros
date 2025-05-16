import cv2
from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints
from typing import List, Dict
def parse_hypothesis(model, results: Results) -> List[Dict]:

    hypothesis_list = []

    if results.boxes:
        box_data: Boxes
        for box_data in results.boxes:
            hypothesis = {
                "class_id": int(box_data.cls),
                "class_name": model.names[int(box_data.cls)],
                "score": float(box_data.conf),
            }
            hypothesis_list.append(hypothesis)

    elif results.obb:
        for i in range(results.obb.cls.shape[0]):
            hypothesis = {
                "class_id": int(results.obb.cls[i]),
                "class_name": model.names[int(results.obb.cls[i])],
                "score": float(results.obb.conf[i]),
            }
            hypothesis_list.append(hypothesis)

    return hypothesis_list

def parse_boxes(results: Results) -> List[BoundingBox2D]:

    boxes_list = []

    if results.boxes:
        box_data: Boxes
        for box_data in results.boxes:

            msg = BoundingBox2D()

            # get boxes values
            box = box_data.xywh[0]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])

            # append msg
            boxes_list.append(msg)

    elif results.obb:
        for i in range(results.obb.cls.shape[0]):
            msg = BoundingBox2D()

            # get boxes values
            box = results.obb.xywhr[i]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.center.theta = float(box[4])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])

            # append msg
            boxes_list.append(msg)

    return boxes_list

def parse_masks(results: Results) -> List[Mask]:

    masks_list = []

    def create_point2d(x: float, y: float) -> Point2D:
        p = Point2D()
        p.x = x
        p.y = y
        return p

    mask: Masks
    for mask in results.masks:

        msg = Mask()

        msg.data = [create_point2d(float(ele[0]), float(ele[1])) for ele in mask.xy[0].tolist()]
        msg.height = results.orig_img.shape[0]
        msg.width = results.orig_img.shape[1]

        masks_list.append(msg)

    return masks_list

def parse_keypoints(threshold,results: Results) -> List[KeyPoint2DArray]:

    keypoints_list = []

    points: Keypoints
    for points in results.keypoints:

        msg_array = KeyPoint2DArray()

        if points.conf is None:
            continue

        for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

            if conf >= threshold:
                msg = KeyPoint2D()

                msg.id = kp_id + 1
                msg.point.x = float(p[0])
                msg.point.y = float(p[1])
                msg.score = float(conf)

                msg_array.data.append(msg)

        keypoints_list.append(msg_array)

    return keypoints_list


def crop_image_from_bbox(cv_image, box):
    center_x = box.center.position.x
    center_y = box.center.position.y
    width = box.size.x
    height = box.size.y

    # Calculate corner coordinates
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)

    padding = 50
    h, w = cv_image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    img = cv_image[y1:y2, x1:x2]
    img =  cv2.resize(img,(640,640))
    #cv_image = cv2.resize(cv_image,(640,640))
    return img