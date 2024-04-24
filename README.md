# YOLO DeepSORT

## 簡介

> 一個簡易包裝過的 Library for YOLOv7 + DeepSORT
>
> 此 Library 有修改過 DeepSORT 的 source code


## 使用範例

### YOLOv7

```python
# detector_example.py
from yolov7_deepsort.detector import YOLOv7_Detector
import cv2
from PIL import Image

if __name__ == "__main__":
    det = YOLOv7_Detector(conf_threshold=0.5)
    det.load_model('dog_cat_best.pt')

    # det.detect('cat_image.jpg')
    # Pass in any image path or Numpy Image using 'BGR' format
    # plot_bb = False output the predictions as [x,y,w,h, confidence, class]
    *first_points, first_conf, first_cls = det.detect('test_image.jpg', plot=False)[0]
    print(det.detect('test_image.jpg'))
    print(first_points, first_conf, first_cls)

    result = det.detect('./test_image.jpg', plot=True)
    cv2.imwrite('../test_image_result.jpg', result)

    if len(result.shape) == 3:  # If it is image, convert it to proper image. detector will give "BGR" image
        result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

```

### YOLOv7 + DeepSORT

```python
# tracker_example.py
from yolov7_deepsort.detector import YOLOv7_Detector
from yolov7_deepsort.tracker import YOLOv7_DeepSORT

if __name__ == '__main__':
    det = YOLOv7_Detector(conf_threshold=0.5)
    det.load_model('dog_cat_best.pt')

    yolo_tracker = YOLOv7_DeepSORT(
        detector=det
    )

    yolo_tracker.track_video('test_video.mp4')


```



