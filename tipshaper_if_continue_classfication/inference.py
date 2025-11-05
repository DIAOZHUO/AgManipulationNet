import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import SPMUtil as spmu
import numpy as np
from pathlib import Path
from tipshaper_if_continue_classfication.model import IZ_Classfier



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = IZ_Classfier().to(device)
model.load_state_dict(torch.load(str(Path(__file__).parent.absolute()) + "/output/model_iz20230523-143802_train_param_iz.pth", map_location=device))
model.eval()
print("     TipShaper If-Continue Classification Model Load Finished!")

def normalize_z(value):
    value = np.array(value) - np.mean(value[:50]) + 0.1
    return np.clip(value, a_min=0, a_max=0.5) * 2

# range: -3.5 - +2 to 0, 1
def normalize_i(value):
    return np.clip((np.array(value) + 3.5) / 5.5, a_min=0, a_max=1)

def inference_by_data(data: spmu.DataSerializer):
    data.load()
    header = spmu.ScanDataHeader.from_dataSerilizer(data)
    if header.Array_Builder == "LazyScanArrayBuilder" and 'ArrayBuilderParam' in data.data_dict:
        z = data.data_dict['ArrayBuilderParam']['scan_result_z']
        z_out = []
        for item in z:
            z_out.extend(item)

        i = data.data_dict['ArrayBuilderParam']['scan_result_i']
        i_out = []
        for item in i:
            i_out.extend(item)

        z = torch.Tensor([normalize_z(z_out)]).to(device)
        i = torch.Tensor([normalize_i(i_out)]).to(device)
        z = z[None, :, :]
        i = i[None, :, :]
        outputs = model(z, i)
        return torch.squeeze(outputs).to('cpu').detach().numpy().copy() > 0.5
    else:
        raise ValueError("Wrong type of DataSerializer.")


def inference_by_path(path):
    return inference_by_data(spmu.DataSerializer(path))


if __name__ == '__main__':
    print(inference_by_path("E:\LabviewProject\Labview-SPMController\python\smart_spm\Datas\si_20230421_tipfix/232_si_20230421"))
