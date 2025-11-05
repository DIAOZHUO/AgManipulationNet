import numpy as np
import SPMUtil as spmu

from ag_manipulation.predict_adsorp import predict_by_data as predict_ag_by_data
from ag_manipulation.predict_cleanHUC import predict_by_data as predict_cleanHUC_by_data


def get_random_ag_atom(data: spmu.DataSerializer):
    try:
        bound_boxes = predict_ag_by_data(data, threshold=0.2)
        random_box = []
        for it in bound_boxes:
            if 30 < it[0] < 226 and 30 < it[1] < 226 and it[5] == 0:
                random_box = it
                break

        if random_box is not None:
            ag_huc_center_pt = ((random_box[0] + random_box[2]) / 2, (random_box[1] + random_box[3]) / 2)
            return ag_huc_center_pt
        else:
            print("can not find ag atom on map")
            return None

    except BaseException as e:
        print(e)
        return None


def get_tip_position(data: spmu.DataSerializer):
    try:
        bound_boxes = predict_cleanHUC_by_data(data, threshold=0.2)
        random_box = []
        ag_huc_center_pt = get_random_ag_atom(data)

        if ag_huc_center_pt is not None:
            for it in bound_boxes:
                huc_center_pt = ((it[0] + it[2]) / 2, (it[1] + it[3]) / 2)
                vec = np.array([ag_huc_center_pt[0] - huc_center_pt[0], ag_huc_center_pt[1] - huc_center_pt[1]])
                distance = np.linalg.norm(vec)
                if 9 < distance < 25:
                    random_box = it
                    break

            if random_box is not None:
                cleanHUC_center_pt = ((random_box[0] + random_box[2]) / 2, (random_box[1] + random_box[3]) / 2)
                tip_position = (
                (ag_huc_center_pt[0] + cleanHUC_center_pt[0]) / 2, (ag_huc_center_pt[1] + cleanHUC_center_pt[1]) / 2)
                return tip_position
            else:
                return None

        else:
            return None

    except BaseException as e:
        print(e)
        return None


if __name__ == '__main__':
    data = spmu.DataSerializer("./511_si_20231108.pkl")
    data.load()

    import matplotlib.pyplot as plt


    plt.imshow(spmu.flatten_map(data.data_dict[spmu.cache_2d_scope.FWFW_ZMap.name], spmu.FlattenMode.Average))

    manipulation_position = get_tip_position(data)

    plt.scatter(manipulation_position[0], manipulation_position[1], color='red', label='Manipulation Position')
    plt.legend()
    plt.show()
