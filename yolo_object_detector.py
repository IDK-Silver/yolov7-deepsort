import random

import typing
import cv2
import numpy
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils.datasets import letterbox
from utils.torch_utils import select_device, TracedModel
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.plots import plot_one_box
from PIL import Image

# for GTX 1650 ti
torch.backends.cudnn.enabled = False


class YOLO_Result:
    def __init__(self):
        self.conf: float = 0.
        self.cls: int = 0
        self.points: typing.List[int, int, int, int] = []

    def decode(self, data: list):
        data = np.array(data)

        if data.ndim >= 2:
            print(f"can't decode list of result")
            exit(-1)

        self.points, self.conf, self.points = data


class YOLO_Detector:
    def __init__(
            self, conf: float = 0.6, conf_threshold: float = 0.25,
            iou_threshold: float = 0.45, classes: list = None
    ):
        self.device = select_device(
            "0" if torch.cuda.is_available()
            else 'cpu'
        )

        self.conf = conf
        self.model: typing.Union[torch.Module, TracedModel, None] = None
        self.image_size = 0
        self.model_classify = None
        self.names: list = []
        self.colors: list = []
        self.half_precision = False
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.stride = None

    def load_model(self, weights: str, image_size: int = 640):

        # whether device can use half precision or not
        self.half_precision = self.device.type != 'cpu'

        # load pytorch model to map_location device
        self.model = attempt_load(weights, map_location=self.device)

        # load image size
        self.stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(image_size, s=self.stride)  # check img_size

        # using trace model
        self.model = TracedModel(self.model, self.device, image_size)

        # enable using half float
        # https://pytorch.org/docs/stable/generated/torch.Tensor.half.html
        if self.half_precision:
            self.model.half()

        # predict once, check that pytorch model can run current
        if self.half_precision:
            # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward
            # model(input) is model predict
            self.model(
                torch.zeros(1, 3, self.image_size, self.image_size).to(self.device)
                .type_as(next(self.model.parameters()))
            )

            # model(input) = model.forward(input)
            # self.model.forward(
            #     torch.zeros(1, 3, self.image_size, self.image_size).to(self.device)
            # )

        # second stage classifier, coming soon
        # YOLO classify https://blog.csdn.net/qq_38253797/article/details/119214728
        # ResNet https://medium.com/@rossleecooloh/直觀理解resnet-簡介-觀念及實作-python-keras-8d1e2e057de2

        # for get classname of train set
        # hasattr(object,name) in python checks
        # if some object has an attribute with 'name'
        # in this case, self.model's type is TraceModel, not has attribute with module
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # random color for box
        self.colors = [
            [random.randint(0, 255) for _ in range(3)]
        ]

    @torch.no_grad()
    def detect(self, image: typing.Union[str, numpy.array], plot: bool = False):

        # load image
        image, original_image = self.load_image(image)
        image: torch.Tensor = torch.from_numpy(image).to(self.device)

        # choose precision
        if self.half_precision:
            image = image.half()
        else:
            image = image.float()

        # normalization, 0 - 255 to 0.0 - 1.0
        image /= 255.0

        # if load image is single, extend dimension
        # https://clay-atlas.com/blog/2020/09/02/pytorch-cn-squeeze-unsqueeze-usage/
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        # predict image
        predict = self.model(image, augment=False)[0]

        predict = non_max_suppression(
            predict, self.conf_threshold, self.iou_threshold,
            classes=self.classes
        )

        detection = predict[0]

        if len(detection):
            # rescale boxes from img_size to im0 size
            # https://blog.csdn.net/m0_46483236/article/details/123605527
            detection[:, :4] = scale_coords(
                image.shape[2:], detection[:, :4], original_image.shape
            ).round()

            for *points, conf, cls in reversed(detection):
                if plot:
                    label = f'{self.names[int(cls)]} {conf:.2f}'

                    plot_one_box(
                        points, original_image, label=label,
                        color=self.colors[int(cls)], line_thickness=2
                    )

            if plot:
                return original_image
            else:
                return detection.detach().cpu().numpy()

        return original_image if plot else None

    def load_image(self, original_image: typing.Union[str, numpy.array]):

        # if image is file path, loading image
        if isinstance(original_image, str):
            original_image = cv2.imread(original_image)
        assert original_image is not None, 'Image Not Found '

        # letterbox https://medium.com/mlearning-ai/letterbox-in-object-detection-77ee14e5ac46
        image: numpy.ndarray = letterbox(
            original_image,
            self.image_size,
            stride=self.stride
        )[0]

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # to 3 x width x height
        # https://zhuanlan.zhihu.com/p/61203757
        image = image.transpose(2, 0, 1)

        # make matrix is continuous in memory
        # https://zhuanlan.zhihu.com/p/59767914
        image = np.ascontiguousarray(image)

        return image, original_image


if __name__ == "__main__":
    det = YOLO_Detector()
    det.load_model('dog_cat_best.pt')
    # det.detect('cat_image.jpg')
    # Pass in any image path or Numpy Image using 'BGR' format
    # plot_bb = False output the predictions as [x,y,w,h, confidence, class]
    result = det.detect('./example/image/cat_image.jpg', plot=True)

    *points, conf, cls = det.detect('./example/image/cat_image.jpg', plot=False)[0]

    print(points, conf, cls)

    cv2.imwrite('cat.jpg', result)

    if len(result.shape) == 3:  # If it is image, convert it to proper image. detector will give "BGR" image
        result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
