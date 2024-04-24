# YOLO DeepSORT

## 簡介

> 一個簡易包裝過的 Library for YOLO + DeepSORT
>
> 此 Library 有修改過 DeepSORT 的 source code


## 使用範例

### YOLO

```python
# detector_example.py
from yolo_object_detector import YOLO_Detector
import cv2
from PIL import Image

if __name__ == "__main__":
    det = YOLO_Detector()
    det.load_model('dog_cat_best.pt')

    # det.detect('cat_image.jpg')
    # Pass in any image path or Numpy Image using 'BGR' format
    # plot_bb = False output the predictions as [x,y,w,h, confidence, class]
    *first_points, first_conf, first_cls = det.detect('./example/image/cat_image.jpg', plot=False)[0]
    print(det.detect('./example/image/cat_image.jpg'))
    print(first_points, first_conf, first_cls)

    result = det.detect('./example/image/cat_image.jpg', plot=True)
    cv2.imwrite('cat.jpg', result)

    if len(result.shape) == 3:  # If it is image, convert it to proper image. detector will give "BGR" image
        result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
```

### YOLO + DeepSORT

```python
# tracker_example.py
from yolo_object_detector import YOLO_Detector
from yolo_deepsort import YOLO_DeepSORT

if __name__ == '__main__':
    det = YOLO_Detector()
    det.load_model('dog_cat_best.pt')

    yolo_tracker = YOLO_DeepSORT(
        detector=det, reid_model_path='./mars-small128.pb'
    )

    yolo_tracker.track_video('two_cat.mp4')

```



