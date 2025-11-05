from tipshaper_if_continue_classfication.inference import inference_by_path
import SPMUtil as spmu
import matplotlib.pyplot as plt

def plot_tipshaper(path):
    data = spmu.DataSerializer(path)
    data.load()
    stage = spmu.StageConfigure.from_dataSerilizer(data)
    z = data.data_dict['ArrayBuilderParam']['scan_result_z']
    z_out = []
    for item in z:
        z_out.extend(item)

    i = data.data_dict['ArrayBuilderParam']['scan_result_i']
    i_out = []
    for item in i:
        i_out.extend(item)
    fig, axes = plt.subplots(2, 2, figsize=(7, 4))
    axes[0, 0].set_title("scan z")
    axes[0, 0].get_xaxis().set_visible(False)
    axes[0, 0].plot(data.data_dict['ArrayBuilderParam']['scan_info']['scan_array_z'], c="red")
    axes[1, 0].set_title("scan V")
    axes[1, 0].plot(data.data_dict['ArrayBuilderParam']['scan_info']['scan_array_v'] + stage.Sample_Bias, c="red")
    axes[0, 1].set_title("result z")
    axes[0, 1].get_xaxis().set_visible(False)
    axes[0, 1].plot(z_out, c="blue")
    axes[1, 1].set_title("result I")
    axes[1, 1].plot(i_out, c="blue")
    plt.show()


if __name__ == '__main__':
    plot_tipshaper("./1236_si_20241213.pkl")
    print("should continue manipulation:", inference_by_path("./1236_si_20241213.pkl"))

    plot_tipshaper("./1237_si_20241213.pkl")
    print("should continue manipulation:", inference_by_path("./1237_si_20241213.pkl"))