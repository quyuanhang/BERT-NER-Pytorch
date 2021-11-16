import json
import os
import random


def load_datas_from_json(fp):
    data = {}
    with open(fp) as f:
        data = json.load(f)
    return data["result"]

def format_file(fp):
    data_list = load_datas_from_json(fp)
    datas = []
    for data in data_list:
        try:
            content_str = data["content"]
            content_list = [x for x in content_str if x.strip() != ""]
            label_list = ["O"] * len(content_list)
            for span in data["spans"]:
                start = span["start_offset"]
                end = span["end_offset"]
                for i in range(start, end):
                    if label_list[i] != "O":
                        break
                else:
                    label_list[start] = "B-" + span["label"]
                    for i in range(start + 1, end):
                        label_list[i] = "I-" + span["label"]
                    # label_list[end - 1] = "E-" + span["label"]
            datas.append([content_list, label_list])
        except:
            print(data)
    return datas

def format_files(input_dir, output_dir):
    train_name = os.path.join(output_dir, "train.txt")
    dev_name = os.path.join(output_dir, "dev.txt")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if os.path.exists(train_name):
        os.remove(train_name)
    if os.path.exists(dev_name):
        os.remove(dev_name)
    for root, _, files in os.walk(input_dir):
        for file in files:
            fp = os.path.join(root, file)
            datas = format_file(fp)
            train_file = open(train_name, mode="a")
            dev_file = open(dev_name, mode="a")
            for content_list, label_list in datas:
                if random.random() > 0.7:
                    f = dev_file
                else:
                    f = train_file
                for content, label in zip(content_list, label_list):
                    f.write("{} {}\n".format(content, label))
                f.write("\n")
            train_file.close()
            dev_file.close()
            
    return


if __name__ == "__main__":
    # format_files("/home/aipf/work/建行杯数据集/舆情预警/train", "/home/aipf/work/建行杯数据集/舆情预警/train2")
    format_files("datasets/aipf", "datasets/aipf2")
