import torch
import numpy as np
import SPMUtil as spmu
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from topo_classification_net.model import CustomConvNet
from enum import Enum

import sys
from pathlib import Path
from typing import Union



@torch.no_grad()
class CNNInference(object):
    def __init__(self, class_num=15, model_directory=""):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        depo_path = str(Path(__file__).parent.absolute())
        model_path = depo_path + model_directory
        self.model = CustomConvNet(class_num).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("     Model Load Finished!")

        self.transforms = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


    def detect_by_img(self, img: np.ndarray, process=True):
        if len(img.shape) == 1:
            raise ValueError("image shape error: input should have 2 or 3 dimension")
        if process:
            img = spmu.flatten_map(img, flatten=spmu.FlattenMode.Average)
            img = gaussian_filter(img, sigma=1)
        img = np.array(spmu.formula.normalize01_2dmap(img) * 255, dtype=np.float32)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = self.transforms(image=img)['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            result = self.model(img)
            # print("result array for all clusters:", result.data)
            # result = torch.divide(result, torch.sum(result))
            # print("output:", torch.max(result.data, 1)[1].item())
            index = torch.max(result.data, 1)[1].item()
            return index, result.to('cpu').detach().numpy()[0].copy()


    def detect_by_path(self, img_path):
        map = np.array(plt.imread(img_path))[:,:,0]
        return self.detect_by_img(map)


    def analyze_result(self, index, accuracy):
        # remove step tip: 15 elements with step 0
        result_list = []
        for i in range(accuracy.shape[0]):
            if accuracy[i] < 3.2:
                result_list.append(0)
            else:
                result_list.append(accuracy[i])
        result_list = np.array(result_list)

        if sum(result_list) == 0:
            return index, 0, False, 0
        return index, result_list[index]/np.sum(result_list), True, np.sum(result_list[:6]) / np.sum(result_list)



def predict_by_map(map):
    model_path = "./model20241127_043417.pth"
    inference = CNNInference(class_num=5, model_directory=model_path)
    result = inference.analyze_result(*inference.detect_by_img(map))
    return result

# class TipQualityType(Enum):
#     Good_Best = 0
#     Good = 1
#     Good_Fat = 2
#     Good_Multi = 3
#     Good_bad_area = 4
#     Good_TipChange = 5
#     Step = 6
#     Bad_Noisy = 7
#     Bad_TipChange = 8
#     Bad_Multi = 9
#     Bad_Multi2 = 10
#     Bad_Multi3 = 11
#     Bad_Area = 12
#     Bad_NotClear = 13
#     Bad_Fail = 14


class TipQualityType(Enum):
    Good = 0
    Bad_Area = 1
    TipChange = 2
    Noisy = 3
    Bad_Tip = 4

if __name__ == '__main__':
    model_path = "./model20241127_043417.pth"
    inference = CNNInference(class_num=5, model_directory=model_path)

    path = "C:\\Users\\HatsuneMiku\\Desktop\\okuyama\\Datas\\si_20240619\\080_si_20240619.pkl"
    # path = "E:\LabviewProject\Labview-SPMController\python\smart_spm\Datas/si_20231115okuyama/546_si_20231115.pkl"
    data = spmu.DataSerializer(path)
    data.load()
    map = data.data_dict[spmu.cache_2d_scope.FWFW_ZMap.name]
    result = inference.analyze_result(*inference.detect_by_img(map))
    print(result)

    tip_quality = TipQualityType(int(result[0]))
    text = "tip type - " + tip_quality.name + " in " + str(result[1]*100) + "%, "
    if result[2]:
        print(text+"bad good judge in " + str(result[3]*100) + "%")
    else:
        print("not clear tip", text)

    img = spmu.flatten_map(data.data_dict[spmu.cache_2d_scope.FWFW_ZMap.name])
    plt.imshow(map, cmap="afmhot")
    plt.axis("off")
    plt.show()


