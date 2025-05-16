# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import torch
from ultralytics import YOLO, YOLOWorld
from ultralytics.engine.results import Results
import numpy as np
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CompressedImage
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import SetClasses
from .utils import *


class YoloNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("yolo_node")

        # params
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("device", "cuda:0")

        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)
        self.declare_parameter("is_compressed", False)
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        self.type_to_model = {"YOLO": YOLO, "World": YOLOWorld}

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # model params
        self.model_type = self.get_parameter("model_type").get_parameter_value().string_value
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value

        # inference params
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        self.retina_masks = self.get_parameter("retina_masks").get_parameter_value().bool_value
        self.is_compressed = self.get_parameter("is_compressed").get_parameter_value().bool_value
        # ros params
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value

        # detection pub
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub = self.create_lifecycle_publisher(DetectionArray, "detections", 10)
        self._lp_pub = self.create_publisher(CompressedImage, "/lp_image/compressed", self.image_qos_profile)
        self.cv_bridge = CvBridge()

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        try:
            self.yolo = self.type_to_model[self.model_type](self.model)
            self.lp_model = YOLO("/root/ros2_ws/src/license.engine")  # TODO
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exists")
            return TransitionCallbackReturn.ERROR

        if self.model.endswith(".engine") is False:
            try:
                self.get_logger().info("Trying to fuse model...")
                self.yolo.fuse()
            except TypeError as e:
                self.get_logger().warn(f"Error while fuse: {e}")

        self._enable_srv = self.create_service(SetBool, "enable", self.enable_cb)

        if isinstance(self.yolo, YOLOWorld):
            self._set_classes_srv = self.create_service(SetClasses, "set_classes", self.set_classes_cb)

        self._sub = self.create_subscription(
            CompressedImage if self.is_compressed else Image, "image_raw", self.image_cb, self.image_qos_profile
        )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        del self.yolo
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()

        self.destroy_service(self._enable_srv)
        self._enable_srv = None

        if isinstance(self.yolo, YOLOWorld):
            self.destroy_service(self._set_classes_srv)
            self._set_classes_srv = None

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._pub)

        del self.image_qos_profile

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def enable_cb(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        self.enable = request.data
        response.success = True
        return response

    def image_cb(self, msg) -> None:

        if self.enable:

            # convert image + predict
            if self.is_compressed:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                del np_arr
            else:
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.yolo.track(
                persist=True,
                # classes=[0,41],
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                iou=self.iou,
                imgsz=(self.imgsz_height, self.imgsz_width),
                half=self.half,
                max_det=self.max_det,
                augment=self.augment,
                agnostic_nms=self.agnostic_nms,
                retina_masks=self.retina_masks,
                device=self.device,
            )
            results: Results = results[0].cpu()

            if results.boxes or results.obb:
                hypothesis = parse_hypothesis(self.yolo, results)
                boxes = parse_boxes(results)

            if results.masks:
                masks = parse_masks(results)

            if results.keypoints:
                keypoints = parse_keypoints(self.threshold, results)

            # create detection msgs
            detections_msg = DetectionArray()

            for i in range(len(results)):

                aux_msg = Detection()

                if results.boxes or results.obb and hypothesis and boxes:
                    # CHeck if it is a vehicle class [2,3,5]
                    if hypothesis[i]["class_id"] in [2, 3, 5]:
                        img = crop_image_from_bbox(cv_image, boxes[i])
                        self._lp_pub.publish(self.cv_bridge.cv2_to_compressed_imgmsg(img))
                        del img
                    aux_msg.class_id = hypothesis[i]["class_id"]
                    aux_msg.class_name = hypothesis[i]["class_name"]
                    aux_msg.score = hypothesis[i]["score"]

                    aux_msg.bbox = boxes[i]

                if results.masks and masks:
                    aux_msg.mask = masks[i]

                if results.keypoints and keypoints:
                    aux_msg.keypoints = keypoints[i]

                detections_msg.detections.append(aux_msg)

            # publish detections
            detections_msg.header = msg.header
            self._pub.publish(detections_msg)

            del results
            del cv_image

    def set_classes_cb(
        self,
        req: SetClasses.Request,
        res: SetClasses.Response,
    ) -> SetClasses.Response:
        self.get_logger().info(f"Setting classes: {req.classes}")
        self.yolo.set_classes(req.classes)
        self.get_logger().info(f"New classes: {self.yolo.names}")
        return res


def main():
    rclpy.init()
    node = YoloNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
