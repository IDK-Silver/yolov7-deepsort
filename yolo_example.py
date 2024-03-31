from yolo_object_detector import YOLO_Detector
import cv2
from PIL import Image

if __name__ == "__main__":
    det = YOLO_Detector()
    det.load_model('dog_cat_best.pt')

    # det.detect('cat_image.jpg')
    # Pass in any image path or Numpy Image using 'BGR' format
    # plot_bb = False output the predictions as [x,y,w,h, confidence, class]
    *first_points, first_conf, first_cls = det.detect('./who_test.jpg', plot=False)[0]
    print(det.detect('./who_test.jpg'))
    print(first_points, first_conf, first_cls)

    result = det.detect('./who_test_3.jpg', plot=True)
    cv2.imwrite('who_3.jpg', result)

    if len(result.shape) == 3:  # If it is image, convert it to proper image. detector will give "BGR" image
        result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
