
from settings import Settings
from utils.file_and_folder_operations import *
import random
import shutil
from multiprocessing import Pool


def create_lists_from_dataset(base_folder):
    lists = []
    # base_folder = /home/yzhang14/Data/MSD/Task03_Liver
    json_file = join(base_folder, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        img_files = d['training']

    for tr in img_files:
        lists.append(join(base_folder, "imagesTr", tr['image'].split("/")[-1]))
        # list = [["liver_14.nii.gz", ...]
    return lists


def split_nifti(files_name, output_dirs):
    file_base = files_name.split("/")[-1]
    shutil.copy(files_name, join(output_dirs, file_base))


def split_dataset(task_string):
    base_folder = join(raw_dataset_dir, task_string)
    files_name = []
    output_dirs = []

    files = create_lists_from_dataset(base_folder)
    files_num = len(files)
    train_num = int(0.6 * files_num)
    val_num = int(0.8 * files_num)
    random.shuffle(files)

    data_file_dict = {"train": files[:train_num],
                      "val": files[train_num:val_num],
                      "test": files[val_num:]}
    print("The length of training data of", task_string,
          "task is", len(data_file_dict["train"]))
    print("The length of validation data of", task_string,
          "task is", len(data_file_dict["val"]))
    print("The length of test data of", task_string,
          "task is", len(data_file_dict["test"]))

    p = Pool(8)
    for phase in ["train", "val", "test"]:
        print("Spliting data of phase:", phase, "......start.")
        folder = join(base_folder, phase)
        if not isdir(folder):
            os.mkdir(folder)

        nii_files = data_file_dict[phase]
        nii_files.sort()
        output_dir = join(base_folder, phase)
        for n in nii_files:
            files_name.append(n)
            output_dirs.append(output_dir)

        p.starmap(split_nifti, zip(files_name, output_dirs))
        print("Spliting data of phase:", phase, "......the end.")
    p.close()
    p.join()

    return data_file_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, help="task name. There must be a matching folder in "
                                                       "raw_dataset_dir", required=True)
    args = parser.parse_args()
    task = args.task

    settings = Settings()
    common_params, net_params, train_params, eval_params = settings['COMMON'], settings['NETWORK'], \
                                                           settings['TRAINING'], settings['EVAL']
    raw_dataset_dir = common_params["data_dir"]
    task_dataset_dict = {}
    task_sets = ["Task03_Liver", "Task06_Lung", "Task07_Pancreas", "Task09_Spleen", "Task10_Colon"]

    if task == "all":
        for tsk in task_sets:
            print("Splitting task:", tsk)
            data_file_dict = split_dataset(tsk)
            task_dataset_dict[tsk] = data_file_dict

    else:
        print("Splitting task:", task)
        data_file_dict = split_dataset(task)
        task_dataset_dict[task] = data_file_dict

