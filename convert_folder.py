import json
import os
import shutil
import random


# Convert label to idx
with open("labels.txt", "r") as f:
    classes = [c.strip() for c in f.readlines()]
    idx_dict = {c: str(i) for i, c in enumerate(classes)}

def convert_labelmes_to_yolo(labelme_folder, output_folder):
    label_folder = os.path.join(output_folder, "labels")
    os.makedirs(label_folder, exist_ok=True)
    image_folder = os.path.join(output_folder, "images")
    os.makedirs(image_folder, exist_ok=True)

    for root, dirs, files in os.walk(labelme_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file)[-1] != ".json":
                shutil.copy(file_path, image_folder)
                print(f"Copied {file_path} to {image_folder}")
            else:
                with open(file_path, 'r') as f:
                    labelme_data = json.load(f)

                image_filename = labelme_data["imagePath"]
                image_width = labelme_data["imageWidth"]
                image_height = labelme_data["imageHeight"]

                txt_filename = os.path.splitext(image_filename)[0] + ".txt"
                txt_path = os.path.join(label_folder, txt_filename)

                with open(txt_path, 'w') as f:
                    for shape in labelme_data["shapes"]:
                        label = shape["label"]
                        points = shape["points"]

                        class_idx = idx_dict.get(label)
                        if class_idx is not None:
                            f.write(class_idx)

                            for point in points:
                                x = point[0] / image_width
                                y = point[1] / image_height
                                f.write(f" {x} {y}")

                            f.write("\n")

                print(f"Converted {file} to {txt_path}")

    print("转换成功")

def split_data(output_folder, dataset_folder):
    random.seed(0)
    split_rate = 0.1
    origin_label_path = os.path.join(output_folder, "labels")
    origin_image_path = os.path.join(output_folder, "images")
    train_label_path = os.path.join(dataset_folder, "train", "labels")
    os.makedirs(train_label_path, exist_ok=True)
    train_image_path = os.path.join(dataset_folder, "train", "images")
    os.makedirs(train_image_path, exist_ok=True)
    val_label_path = os.path.join(dataset_folder, "val", "labels")
    os.makedirs(val_label_path, exist_ok=True)
    val_image_path = os.path.join(dataset_folder, "val", "images")
    os.makedirs(val_image_path, exist_ok=True)

    images = os.listdir(origin_image_path)
    num = len(images)
    eval_index = random.sample(images,k=int(num*split_rate))
    for single_image in images:
        origin_single_image_path = os.path.join(origin_image_path, single_image)
        single_txt = os.path.splitext(single_image)[0] + ".txt"
        origin_single_txt_path = os.path.join(origin_label_path, single_txt)
        if single_image in eval_index:
            #single_json_path = os.path.join(val_label_path,single_json)
            shutil.copy(origin_single_image_path, val_image_path)
            shutil.copy(origin_single_txt_path, val_label_path)
        else:
            #single_json_path = os.path.join(train_label_path,single_json)
            shutil.copy(origin_single_image_path, train_image_path)
            shutil.copy(origin_single_txt_path, train_label_path)

    print("数据集划分完成")

    with open(os.path.join(dataset_folder,"data.yaml"),"w") as f:
        f.write(f"train: {dataset_folder}\n")
        f.write(f"val: {val_image_path}\n")
        f.write(f"test: {val_image_path}\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")


labelme_folder = "labelme_folder" #labelme生成的标注文件所在的文件夹
output_folder = "output_dir" #存储yolo标注文件的文件夹
dataset_folder = "dataset" #存储划分好的数据集的文件夹
convert_labelmes_to_yolo(labelme_folder, output_folder)#将labelme标注文件转换为yolo格式
split_data(output_folder, dataset_folder)#划分训练集和验证级
