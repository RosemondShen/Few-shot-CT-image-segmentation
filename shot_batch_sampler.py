import numpy as np
from settings import Settings
from utils.file_and_folder_operations import *
import os
import SimpleITK as sitk
import torch
import random
import matplotlib.pyplot as plt
import copy
import torch.utils.data as data

settings = Settings()
common_params, net_params, train_params, eval_params = settings['COMMON'], settings['NETWORK'], \
                                                           settings['TRAINING'], settings['EVAL']
lab_list_fold = common_params["lab_list_fold"]
data_dir = os.listdir("/home/yzhang14/Data/MSD")
'''
lab_list_fold = {"fold1": {"train": ["Liver00", "Liver01", "Lung00", "Pancreas00", "Pancreas01", "Spleen00"], "test": ["Colon00"]},
                 "fold2": {"train": ["Liver01", "Lung00", "Pancreas00", "Pancreas01", "Spleen00", "Colon00"], "test": ["Liver00"]},
                 "fold3": {"train": ["Liver00", "Lung00", "Pancreas00", "Pancreas01", "Spleen00", "Colon00"], "test": ["Liver01"]},
                 "fold4": {"train": ["Liver00", "Liver01", "Pancreas00", "Pancreas01", "Spleen00", "Colon00"], "test": ["Lung00"]},
                 "fold5": {"train": ["Liver00", "Liver01", "Lung00", "Pancreas01", "Spleen00", "Colon00"], "test": ["Pancreas00"]},
                 "fold6": {"train": ["Liver00", "Liver01", "Lung00", "Pancreas00", "Spleen00", "Colon00"], "test": ["Pancreas01"]},
                 "fold7": {"train": ["Liver00", "Liver01", "Lung00", "Pancreas00", "Pancreas01", "Colon00"], "test": ["Spleen00"]},}
'''


def get_lab_list(fold, phase):
    if phase == "val":
        phase = "train"
    return lab_list_fold[fold][phase]


def get_dirs(label, phase):
    organ = label[:-2]
    folder = [file for file in data_dir if file.find(organ) != -1]
    if phase == "train":
        train_dir = join(common_params["data_dir"], folder[0], "train")
        label_dir = join(common_params["data_dir"], folder[0], "labelsTr")
        return train_dir, label_dir
    elif phase == "val":
        val_dir = join(common_params["data_dir"], folder[0], "val")
        label_dir = join(common_params["data_dir"], folder[0], "labelsTr")
        return val_dir, label_dir
    else:
        test_dir = join(common_params["data_dir"], folder[0], "test")
        label_dir = join(common_params["data_dir"], folder[0], "labelsTr")
        return test_dir, label_dir


def get_ids(label, phase):
    img_dir, _ = get_dirs(label, phase)
    ids = [file[:-7] for file in os.listdir(img_dir)]
    return ids


def get_image_num(label, phase):
    ids = get_ids(label, phase)
    file_num = len(ids)
    return file_num


def get_image_num_dict(lab_list, phase):
    num_list = {label: get_image_num(label, phase) for label in lab_list}
    return num_list


def gen_query_label(lab_list, prob):
    """
    Returns a query label uniformly from the label list of current phase.
    :return: random query label
    """
    lab_num = len(lab_list)
    query_label_id = np.random.choice(range(lab_num), size=1, p=prob)
    query_label = lab_list[int(query_label_id)]
    return query_label


def get_img_label_array(query_img_num, query_label, phase):

    train_id = np.random.choice(range(query_img_num), size=1)
    ids = get_ids(query_label, phase)
    train_id = ids[int(train_id)]

    train_dir, label_dir = get_dirs(query_label, phase)
    train_img = join(train_dir, train_id + ".nii.gz")
    train_label = join(label_dir, train_id + ".nii.gz")

    img_itk = sitk.ReadImage(train_img)
    img_array = sitk.GetArrayFromImage(img_itk).astype(np.float32)

    label_itk = sitk.ReadImage(train_label)
    label_array = sitk.GetArrayFromImage(label_itk).astype(np.float32)

    return img_array, label_array


def get_img_label_slice(num_slices, img_array, label_array):
    slice_ids = np.random.choice(range(num_slices), size=2)
    support_img = img_array[slice_ids[0], :, :]
    support_label = label_array[slice_ids[0], :, :]
    qry_img = img_array[slice_ids[1], :, :]
    qry_label = label_array[slice_ids[1], :, :]

    return support_img, support_label, qry_img, qry_label


def normalize_hu(img):

    # lower_bound = np.percentile(img, 99.5)
    # upper_bound = np.percentile(img, 0.5)

    img_std = np.std(img)
    img_mean = np.mean(img)

    # img = np.clip(img, lower_bound, upper_bound)
    img = (img - img_mean) / img_std
    '''
    img_min = np.min(img)
    img_max = np.max(img)
    img = (img - img_min) / (img_max - img_min)
    '''

    return img

def get_image_and_masks(lab_list, phase):
    all_img_masks = []
    lab_list = ["Liver00"]

    for label in lab_list:
        # label = "Liver00", "", "", ......
        print("The splitting label is:", label, "......")
        if label != "Liver01" and label != "Pancreas01":

            img_masks = []
            ids = get_ids(label, phase)
            random.shuffle(ids)
            img_dir, label_dir = get_dirs(label, phase)
            for id in ids:
                img_file = join(img_dir, id + ".nii.gz")
                mask_file = join(label_dir, id + ".nii.gz")

                img_itk = sitk.ReadImage(img_file)
                img_array = sitk.GetArrayFromImage(img_itk).astype(np.float32)
                mask_itk = sitk.ReadImage(mask_file)
                mask_array = sitk.GetArrayFromImage(mask_itk).astype(np.float32)

                imgmask = []

                for c in range(mask_array.shape[0]):
                    mask = mask_array[c, :, :]
                    img = img_array[c, :, :]

                    mask[mask == 2] = 0
                    area1 = np.sum(mask)

                    if label == "Liver00":
                        if 1 in np.unique(mask) and area1 > 40000:

                            img = normalize_hu(img)
                            imgmask.append([img, mask])
                            if phase == "train" and len(imgmask) >= 4:
                                print("Until now, the selected slices number are:", len(imgmask))
                                break
                            elif phase == "val" and len(imgmask) >= 4:
                                print("Until now, the selected slices number are:", len(imgmask))
                                break
                
                if len(imgmask) % 2 == 1:
                    imgmask.pop()
                img_masks.extend(imgmask)
                if phase == "train" and len(img_masks) >= 4:
                    break
                elif phase == "val" and len(img_masks) >= 4:
                    break
            all_img_masks.extend(img_masks)
            print("Until now, the selected slices of", phase, "of ", label,  " number are:", len(all_img_masks))

        print("The splitting of ", label, " is done.")
    return InputData(all_img_masks)


def batch(iterable, batch_size):
    b = []
    for i, img_mask in enumerate(iterable):
        b.append(img_mask)
        if (i+1) % batch_size == 0:
            yield b
            b = []


class InputData(data.Dataset):
    def __init__(self, x):
        self.X = x

    def __getitem__(self, index):

        input1 = torch.from_numpy(self.X[index][0]).type(torch.FloatTensor)
        input1 = input1[np.newaxis, :, :]
        y1 = torch.from_numpy(self.X[index][1]).type(torch.FloatTensor)
        y1 = y1[np.newaxis, :, :]
        return input1, y1  # , input2, y2

    def __len__(self):
        # return int(len(self.X)/2)
        return int(len(self.X))




