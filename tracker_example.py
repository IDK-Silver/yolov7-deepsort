from yolo_object_detector import YOLO_Detector
from yolo_deepsort import YOLO_DeepSORT

if __name__ == '__main__':
    det = YOLO_Detector()
    det.load_model('dog_cat_best.pt')

    yolo_tracker = YOLO_DeepSORT(
        detector=det, reid_model_path='./mars-small128.pb'
    )

    yolo_tracker.track_video('two_cat.mp4')
