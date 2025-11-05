import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import SPMUtil as spmu
import cv2
import copy
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure






class CustomPklDataset(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                if os.path.splitext(img_file)[1] == ".pkl":
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    @staticmethod
    def contrast_stretching(m, percent=1):
        p2, p98 = np.percentile(m, (percent, 100 - percent))
        return exposure.rescale_intensity(m, in_range=(p2, p98))

    def __getitem__(self, index):
        data = spmu.DataSerializer(self.image_files_path[index])
        data.load()
        map = spmu.flatten_map(data.data_dict[spmu.cache_2d_scope.FWFW_ZMap.name], flatten=spmu.FlattenMode.Average)
        img = gaussian_filter(map, sigma=1)
        # img = self.contrast_stretching(img, percent=1)
        img = np.array(spmu.formula.normalize01_2dmap(img) * 255, dtype=np.float32)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        return {'image': img, 'label': self.labels[index]}

    def __len__(self):
        return self.length


class CustomImageDataset(Dataset):
    def read_data_set(self): # trainフォルダ内のpngファイルなどを取得

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                if os.path.splitext(img_file)[1] == ".png":
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None, transforms_tip_change=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms
        self.transforms_tip_change = transforms_tip_change

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = np.asarray(image.convert("RGB"), dtype=np.float32)
        label = self.labels[index]
        # print(label, self.image_files_path[index])
        if label == 2:
            if self.transforms_tip_change is not None:
                image = self.transforms_tip_change(image=image)["image"]
        else:
            if self.transforms is not None:
                image = self.transforms(image=image)["image"]

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length




class SiameseImageDataset(Dataset):

    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                if os.path.splitext(img_file)[1] == ".png":
                    all_img_files.append(img_file)
                    all_labels.append(label)
        return all_img_files, all_labels, len(all_img_files), len(class_names)


    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        # self.classes = data_set_path.classes
        # self.imgs = data_set_path.imgs
        self.transform = transforms


    def __getitem__(self, index):
        # CHOOSE EITHER POSITIVE PAIR (0) OR NEGATIVE PAIR (1)
        self.target = np.random.randint(0, 2)
        # HERE THE FIRST IMAGE IS CHOSEN BY VIRTUE OF INDEX ITSELF
        img1 = self.image_files_path[index]
        label1 = self.labels[index]
        # CREATE NEW LIST OF IMAGES TO AVOID RE-SELECTING ORIGINAL IMAGE
        new_labels = self.labels.copy()
        new_labels.pop(index)
        new_imgs_path = list(set(self.image_files_path) - set(self.image_files_path[index]))
        length = len(new_labels)
        # print(length)
        random = np.random.RandomState(42)
        if self.target == 1:
            # GET NEGATIVE COUNTERPART
            label2 = label1
            while label2 == label1:
                choice = random.choice(length)
                img2 = new_imgs_path[choice]
                label2 = new_labels[choice]
        else:
            # GET POSITIVE COUNTERPART
            label2 = label1 + 1
            while label2 != label1:
                choice = random.choice(length)
                img2 = new_imgs_path[choice]
                label2 = new_labels[choice]

        img1 = np.asarray(Image.open(img1).convert('RGB'), dtype=np.float32)
        img2 = np.asarray(Image.open(img2).convert('RGB'), dtype=np.float32)
        if self.transform:
            img1 = self.transform(image=img1)
            img2 = self.transform(image=img2)
        return img1, img2, self.target

    def __len__(self):
        return self.length

