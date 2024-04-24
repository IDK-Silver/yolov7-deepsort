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
