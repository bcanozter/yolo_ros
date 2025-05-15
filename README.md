# yolo_ros

ROS 2 wrap for YOLO models from [Ultralytics](https://github.com/ultralytics/ultralytics) to perform object detection and tracking, instance segmentation, human pose estimation and Oriented Bounding Box (OBB). There are also 3D versions of object detection, including instance segmentation, and human pose estimation based on depth images.

## Models

The compatible models for yolo_ros are the following:

- [YOLOv3](https://docs.ultralytics.com/models/yolov3/)
- [YOLOv4](https://docs.ultralytics.com/models/yolov4/)
- [YOLOv5](https://docs.ultralytics.com/models/yolov5/)
- [YOLOv6](https://docs.ultralytics.com/models/yolov6/)
- [YOLOv7](https://docs.ultralytics.com/models/yolov7/)
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [YOLOv9](https://docs.ultralytics.com/models/yolov9/)
- [YOLOv10](https://docs.ultralytics.com/models/yolov10/)
- [YOLOv11](https://docs.ultralytics.com/models/yolo11/)
- [YOLOv12](https://docs.ultralytics.com/models/yolo12/)
- [YOLO-World](https://docs.ultralytics.com/models/yolo-world/)


### Topics

- **/yolo/detections**: Objects detected by YOLO using the RGB images. Each object contains a bounding box and a class name. It may also include a mark or a list of keypoints.
- **/yolo/tracking**: Objects detected and tracked from YOLO results. Each object is assigned a tracking ID.
- **/yolo/detections_3d**: 3D objects detected. YOLO results are used to crop the depth images to create the 3D bounding boxes and 3D keypoints.
- **/yolo/debug_image**: Debug images showing the detected and tracked objects. They can be visualized with rviz2.

### Parameters

These are the parameters from the [yolo.launch.py](./yolo_bringup/launch/yolo.launch.py), used to launch all models. Check out the [Ultralytics page](https://docs.ultralytics.com/modes/predict/#inference-arguments) for more details.

- **model_type**: Ultralytics model type (default: YOLO)
- **model**: YOLO model (default: yolov8m.pt)
- **tracker**: tracker file (default: bytetrack.yaml)
- **device**: GPU/CUDA (default: cuda:0)
- **enable**: whether to start YOLO enabled (default: True)
- **threshold**: detection threshold (default: 0.5)
- **iou**: intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS) (default: 0.7)
- **imgsz_height**: image height for inference (default: 480)
- **imgsz_width**: image width for inference (default: 640)
- **half**: whether to enable half-precision (FP16) inference speeding up model inference with minimal impact on accuracy (default: False)
- **max_det**: maximum number of detections allowed per image (default: 300)
- **augment**: whether to enable test-time augmentation (TTA) for predictions improving detection robustness at the cost of speed (default: False)
- **agnostic_nms**: whether to enable class-agnostic Non-Maximum Suppression (NMS) merging overlapping boxes of different classes (default: False)
- **retina_masks**: whether to use high-resolution segmentation masks if available in the model, enhancing mask quality for segmentation (default: False)
- **input_image_topic**: camera topic of RGB images (default: /camera/rgb/image_raw)
- **image_reliability**: reliability for the image topic: 0=system default, 1=Reliable, 2=Best Effort (default: 1)
- **input_depth_topic**: camera topic of depth images (default: /camera/depth/image_raw)
- **depth_image_reliability**: reliability for the depth image topic: 0=system default, 1=Reliable, 2=Best Effort (default: 1)
- **input_depth_info_topic**: camera topic for info data (default: /camera/depth/camera_info)
- **depth_info_reliability**: reliability for the depth info topic: 0=system default, 1=Reliable, 2=Best Effort (default: 1)
- **target_frame**: frame to transform the 3D boxes (default: base_link)
- **depth_image_units_divisor**: divisor to convert the depth image into meters (default: 1000)
- **maximum_detection_threshold**: maximum detection threshold in the z-axis (default: 0.3)
- **use_tracking**: whether to activate tracking after detection (default: True)
- **use_3d**: whether to activate 3D detections (default: False)
- **use_debug**: whether to activate debug node (default: True)

## LICENSE

[![License: MIT](https://img.shields.io/badge/GitHub-GPL--3.0-informational)](https://opensource.org/license/gpl-3-0)

González-Santamarta, M. Á. (2023). yolo_ros [Computer software]. https://github.com/mgonzs13/yolo_ros