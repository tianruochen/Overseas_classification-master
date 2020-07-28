import pandas as pd
import os
from tqdm import tqdm

# pd.set_option("display.max_columns", None)

task = "unpron"
modify_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary", task)


def get_modify_data(mode='val'):
    modify_dir = os.path.join(modify_root_dir, mode)
    modify_files = os.listdir(modify_dir)
    modify_file_list = []
    for modify_file in modify_files:
        modify_file_list.append(os.path.join(modify_dir,modify_file))
    df = pd.DataFrame(columns=("img_labels", "prd_probas", "img_dirs", "modify_cla"))
    for csvfile in modify_file_list:
        temp_df = pd.read_csv(csvfile)
        df = df.append(temp_df,ignore_index=True)
    df = df.dropna(how='any')
    df = df[df["img_labels"] != df["modify_cla"]][["img_dirs", "img_labels", "modify_cla"]]
    modify_data_list = df.values.tolist()
    print(len(modify_data_list))

    modify_data_basepath = "modify_{}_data.csv".format(mode)
    modify_data_path = os.path.join(modify_dir,modify_data_basepath)
    if not os.path.exists(modify_data_path):
        df.to_csv(modify_data_path)
    return modify_data_list


def generate_dataset(modify_data_list, mode):
    new_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    if not os.path.exists(new_dataset_dir):
        os.mkdir(new_dataset_dir)
    if mode == "val":
        origin_dataset = '/data1/wangruihao/cocofun/mass1/val.txt'
        new_dataset = os.path.join(new_dataset_dir,"val.txt")
    else:
        origin_dataset = "/data1/wangruihao/cocofun/mass1/train.txt"
        new_dataset = new_dataset = os.path.join(new_dataset_dir, "train.txt")

    new_data_list = []
    skip_count = 0
    with open(origin_dataset, "r") as f:
        for line in f.readlines():
            img_path,label = line.strip().split("\t")

            skip = False
            for i, path in enumerate([modify_data[0] for modify_data in modify_data_list]):
                if path == img_path:
                    if int(modify_data_list[i][2]) < 0:
                        skip_count += 1
                        skip = True
                        break
                    label = modify_data_list[i][2]
                    break
            if skip:
                continue
            new_data_list.append([img_path, label])
    print("{} skip count: {}".format(mode, skip_count))

    with open(new_dataset, "w") as f:
        for path,new_label in tqdm(new_data_list):
            # print(path + "\t" + str(int(new_label)))
            print(path + "\t" + str(int(new_label)), file=f)
        print("{} data generate done: {}".format(mode, len(new_data_list)))


if __name__ == "__main__":

    for mode in ["train", "val"]:
        modify_data_list = get_modify_data(mode)
        generate_dataset(modify_data_list, mode)
