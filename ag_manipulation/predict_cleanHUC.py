import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import SPMUtil as spmu
import numpy as np
from pathlib import Path
from ultralytics.nn.modules.head import Detect
import torch
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
depo_path = str(Path(__file__).parent.absolute())
MODEL = YOLO(depo_path+"/cleanHUC_best.pt")



def process_map(map):
    map = spmu.flatten_map(map, spmu.FlattenMode.Average)
    map = spmu.filter_2d.gaussian_filter(map, 1)
    map = spmu.formula.normalize01_2dmap(map) * 255
    map = np.stack((map,)*3, axis=-1)
    return map


def _process_predict_result(results, threshold=0.3):
    result = results[0].boxes.data.to('cpu').detach().numpy().copy()
    output = []
    for it in result:
        if it[4] > threshold:
            output.append(it)
    return output

def predict_by_data(data, threshold=0.3):
    if isinstance(data, spmu.DataSerializer):
        data.load()
        results = MODEL(process_map(data.data_dict[spmu.cache_2d_scope.FWFW_ZMap.name]), verbose=False)
    elif isinstance(data, np.ndarray):
        results = MODEL(data)
    else:
        raise ValueError("unsupported input data type", type(data))
    return _process_predict_result(results, threshold)


def predict_by_path(path, threshold=0.3):
    filename, ext = os.path.splitext(path)
    if ext == ".pkl":
        data = spmu.DataSerializer(path)
        return predict_by_data(data, threshold)
    else:
        results = MODEL(path, verbose=False)

    return _process_predict_result(results, threshold)




if __name__ == '__main__':
    """
    plot matrix
    """
    # import util.plot as plot
    # metrics = MODEL.val()
    # plot.plot_matrix(metrics.confusion_matrix, names=["cell type1", "cell type2"])

    # path = "E:\PythonProjects\yolov5-master/tip_quality_classification/asset_type_pklv2/train/01good/313_sn_si_20221007_drift_FWFW_ZMap.png"
    # path = "E:\LabviewProject\Labview-SPMController\python\smart_spm\Datas\si_20230606/878_si_20230606_FWFW_ZMap.png

    # path = "../test_data/079_si_20230521.pkl"
    path = "C:\\Users\\HatsuneMiku\\Desktop\\okuyama\\Datas\\si_20241211\\057_si_20241211.pkl"
    data = spmu.DataSerializer(path)
    data.load()

    map = spmu.flatten_map(data.data_dict[spmu.cache_2d_scope.FWFW_ZMap.name])
    # map = (map - np.min(map)) / (np.max(map) - np.min(map)) * 255
    # akaze = cv2.AKAZE_create()
    # kp1, des1 = akaze.detectAndCompute(map.astype('uint8'), None)
    # for it in kp1:
    #     plt.scatter([it.pt[0]], [it.pt[1]])
    # print(kp1)

    plt.imshow(map, cmap="afmhot")
    # plt.show()
    bound_boxes = predict_by_path(path, threshold=0.5)
    for it in bound_boxes:
        print(it)
        if it[5] == 0:
            spmu.Rect2D((it[0], it[1]), it[2] - it[0], it[3] - it[1]).draw_rect_patch_on_matplot(plt.gca(), "cyan")
            #rect = spmu.Rect2D((it[0], it[1]), it[2] - it[0], it[3] - it[1])
            #plt.scatter([rect.center[0]], [rect.center[1]])
            # plt.imshow(rect.extract_2d_map_from_rect(map))
            # plt.show()

            # rect_map = rect.extract_2d_map_from_rect(map)
            # rect_map = (rect_map - np.min(rect_map)) / (np.max(rect_map) - np.min(rect_map)) * 255
            # akaze = cv2.AKAZE_create()
            # kp1, des1 = akaze.detectAndCompute(rect_map.astype('uint8'), None)
            # print(kp1, des1)

        elif it[5] == 1:
            spmu.Rect2D((it[0], it[1]), it[2] - it[0], it[3] - it[1]).draw_rect_patch_on_matplot(plt.gca(), "r")
    plt.axis("off")
    plt.show()
