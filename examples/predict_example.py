import SPMUtil as spmu
import matplotlib.pyplot as plt
from topo_classification_net.predict import CNNInference, TipQualityType
from ag_manipulation.predict_adsorp import predict_by_data as predict_ag_by_data
from ag_manipulation.predict_cleanHUC import predict_by_data as predict_cleanHUC_by_data

model_path = "./model20241127_043417.pth"
inference = CNNInference(class_num=5, model_directory=model_path)


data = spmu.DataSerializer("./511_si_20231108.pkl")
data.load()
map = data.data_dict[spmu.cache_2d_scope.FWFW_ZMap.name]


"""
topo_classification_net
"""
print("==================")
print("topo_classification_net")
print("==================")
result = inference.analyze_result(*inference.detect_by_img(map))

tip_quality = TipQualityType(int(result[0]))
text = "tip type - " + tip_quality.name + " in " + str(result[1] * 100) + "%, "
if result[2]:
    print(text + "bad good judge in " + str(result[3] * 100) + "%")
else:
    print("not clear tip", text)


"""
cleanHUC detection net
"""
print("==================")
print("cleanHUC detection net")
print("==================")
plt.imshow(map)
bound_boxes = predict_cleanHUC_by_data(data, threshold=0.5)
for it in bound_boxes:
    print(it)
    if it[5] == 0:
        spmu.Rect2D((it[0], it[1]), it[2] - it[0], it[3] - it[1]).draw_rect_patch_on_matplot(plt.gca(), "cyan")
    elif it[5] == 1:
        spmu.Rect2D((it[0], it[1]), it[2] - it[0], it[3] - it[1]).draw_rect_patch_on_matplot(plt.gca(), "r")
plt.title("cleanHUC detection net")
plt.axis("off")
plt.show()


"""
Ag detection net
"""
print("==================")
print("Ag detection net")
print("==================")
plt.imshow(map)
bound_boxes = predict_ag_by_data(data, threshold=0.01)
for it in bound_boxes:
    print(it)
    if it[5] == 0:
        spmu.Rect2D((it[0], it[1]), it[2] - it[0], it[3] - it[1]).draw_rect_patch_on_matplot(plt.gca(), "cyan")
    elif it[5] == 1:
        spmu.Rect2D((it[0], it[1]), it[2] - it[0], it[3] - it[1]).draw_rect_patch_on_matplot(plt.gca(), "r")
    elif it[5] == 2:
        spmu.Rect2D((it[0], it[1]), it[2] - it[0], it[3] - it[1]).draw_rect_patch_on_matplot(plt.gca(), "g")
plt.title("Ag detection net")
plt.axis("off")
plt.show()
