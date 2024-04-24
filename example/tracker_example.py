from yolov7_deepsort.detector import YOLOv7_Detector
from yolov7_deepsort.tracker import YOLOv7_DeepSORT

if __name__ == '__main__':
    det = YOLOv7_Detector(conf_threshold=0.5)
    det.load_model('dog_cat_best.pt')

    yolo_tracker = YOLOv7_DeepSORT(
        detector=det
    )

    yolo_tracker.track_video('test_video.mp4')
