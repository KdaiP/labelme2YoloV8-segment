import json
import random
import yaml
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# 设定随机种子以确保可重复性
random.seed(114514)

# yoloV8支持的图像格式
# https://docs.ultralytics.com/modes/predict/?h=format+image#images
image_formats = ["jpg", "jpeg", "png", "bmp", "webp", "tif", ".dng", ".mpo", ".pfm"]


def copy_labled_img(json_path: Path, target_folder: Path, task: str):
    # 遍历支持的图像格式，查找并复制图像文件
    for format in image_formats:
        image_path = json_path.with_suffix("." + format)
        if image_path.exists():
            # 构建目标文件夹中的目标路径
            target_path = target_folder / "images" / task / image_path.name
            shutil.copy(image_path, target_path)


def json_to_yolo(json_path: Path, sorted_keys: list):
    with open(json_path, "r") as f:
        labelme_data = json.load(f)

    width = labelme_data["imageWidth"]
    height = labelme_data["imageHeight"]
    yolo_lines = []

    for shape in labelme_data["shapes"]:
        label = shape["label"]
        points = shape["points"]
        class_idx = sorted_keys.index(label)
        txt_string = f"{class_idx} "

        for x, y in points:
            x /= width
            y /= height
            txt_string += f"{x} {y} "

        yolo_lines.append(txt_string.strip() + "\n")

    return yolo_lines


def create_directory_if_not_exists(directory_path):
    # 使用 exist_ok=True 可以避免重复检查目录是否存在
    directory_path.mkdir(parents=True, exist_ok=True)

# 创建训练使用的yaml文件
def create_yaml(output_folder: Path, sorted_keys: list):
    train_img_path = Path("images") / "train"
    val_img_path = Path("images") / "val"
    train_label_path = Path("labels") / "train"
    val_label_path = Path("labels") / "val"

    # 创建所需目录
    for path in [train_img_path, val_img_path, train_label_path, val_label_path]:
        create_directory_if_not_exists(output_folder / path)

    names_dict = {idx: name for idx, name in enumerate(sorted_keys)}
    yaml_dict = {
        "path": output_folder.as_posix(),
        "train": train_img_path.as_posix(),
        "val": val_img_path.as_posix(),
        "names": names_dict,
    }

    yaml_file_path = output_folder / "yolo.yaml"
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(yaml_dict, yaml_file, default_flow_style=False, sort_keys=False)

    print(f"yaml created in {yaml_file_path.as_posix()}")


# Convert label to idx
def get_labels_and_json_path(input_folder: Path):
    json_file_paths = list(input_folder.rglob("*.json"))
    label_counts = defaultdict(int)

    for json_file_path in json_file_paths:
        with open(json_file_path, "r") as f:
            labelme_data = json.load(f)
        for shape in labelme_data["shapes"]:
            label = shape["label"]
            label_counts[label] += 1
    
    # 根据标签出现次数排序标签
    sorted_keys = sorted(label_counts, key=lambda k: label_counts[k], reverse=True)
    return sorted_keys, json_file_paths


def labelme_to_yolo(
    json_file_paths: list, output_folder: Path, sorted_keys: list, split_rate: float
):
    # 随机打乱 JSON 文件路径列表
    random.shuffle(json_file_paths)

    # 计算训练集和验证集的分割点
    split_point = int(split_rate * len(json_file_paths))
    train_set = json_file_paths[:split_point]
    val_set = json_file_paths[split_point:]

    for json_file_path in tqdm(train_set):
        txt_name = json_file_path.with_suffix(".txt").name
        yolo_lines = json_to_yolo(json_file_path, sorted_keys)
        output_json_path = Path(output_folder / "labels" / "train" / txt_name)
        with open(output_json_path, "w") as f:
            f.writelines(yolo_lines)
        copy_labled_img(json_file_path, output_folder, task="train")

    for json_file_path in tqdm(val_set):
        txt_name = json_file_path.with_suffix(".txt").name
        yolo_lines = json_to_yolo(json_file_path, sorted_keys)
        output_json_path = Path(output_folder / "labels" / "val" / txt_name)
        with open(output_json_path, "w") as f:
            f.writelines(yolo_lines)
        copy_labled_img(json_file_path, output_folder, task="val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="labelme2yolo")
    parser.add_argument("input_folder", help="输入LabelMe格式文件的文件夹")
    parser.add_argument("output_folder", help="输出YOLO格式文件的文件夹")
    parser.add_argument("split_rate", help="调整训练集和测试集的比重")

    args = parser.parse_args()
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    split_rate = float(args.split_rate)

    sorted_keys, json_file_paths = get_labels_and_json_path(input_folder)
    create_yaml(output_folder, sorted_keys)
    labelme_to_yolo(json_file_paths, output_folder, sorted_keys, split_rate)
