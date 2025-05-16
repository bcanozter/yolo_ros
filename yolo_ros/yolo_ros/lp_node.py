#WIP, MESSY COMMIT

import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.node import Node
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
from sensor_msgs.msg import Image, CompressedImage

from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from paddleocr import PaddleOCR
from datetime import datetime
from .utils import *

#WIP
class LPNode(Node):

    def __init__(self) -> None:
        super().__init__("lp_node")
        self.enable = True
        self.model = "/root/ros2_ws/src/license.engine"
        self.is_compressed = True
        self.threshold = 0.4
        self.device = "cuda:0"
        self.configure()
        self.configure_models()

    def configure(self):
        self.image_qos_profile = QoSProfile(
            reliability=2,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.cv_bridge = CvBridge()

    def configure_models(self):
        try:
            self.lp_model = YOLO(self.model)  # TODO
            self.ocr = PaddleOCR(lang="en",use_gpu=True)
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exists")

        if self.model.endswith(".engine") is False:
            try:
                self.get_logger().info("Trying to fuse model...")
                self.lp_model.fuse()
            except TypeError as e:
                self.get_logger().warn(f"Error while fuse: {e}")

        img_topic_type = CompressedImage if self.is_compressed else Image
        self._sub = self.create_subscription(
            img_topic_type, "/lp_image/compressed", self.image_cb, self.image_qos_profile
        )

    def image_cb(self, msg) -> None:
        if self.enable:
            if self.is_compressed:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                del np_arr
            else:
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            results = self.lp_model(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                half=True,
                device=self.device,
            )
            results: Results = results[0].cpu()
            if results.boxes or results.obb:
                hypothesis = parse_hypothesis(self.lp_model, results)
                boxes = parse_boxes(results)
            detections_msg = DetectionArray()
            for i in range(len(results)):
                aux_msg = Detection()

                if results.boxes or results.obb and hypothesis and boxes:
                    aux_msg.class_id = hypothesis[i]["class_id"]
                    aux_msg.class_name = hypothesis[i]["class_name"]
                    aux_msg.score = hypothesis[i]["score"]

                    aux_msg.bbox = boxes[i]
                    cropped = crop_image_from_bbox(cv_image, boxes[i])
                    # cv2.imwrite(f'/root/ros2_ws/src/dets/lp_{datetime.now()}.jpg',cropped)
                    result = self.ocr.ocr(cropped, det=True, cls=False)
                    for idx in range(len(result)):
                        res = result[idx]
                        if res is not None:
                            for line in res:
                                self.get_logger().info(f"{line[1][0].strip()}")
# [[[155.0, 304.0], [496.0, 321.0], [489.0, 456.0], [148.0, 439.0]], ('NA54KGJ', 0.875269889831543)]
                detections_msg.detections.append(aux_msg)
            # publish detections
            detections_msg.header = msg.header
            # self._pub.publish(detections_msg)

            del results
            del cv_image


def main():
    rclpy.init()
    node = LPNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
