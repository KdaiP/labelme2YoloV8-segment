# labelme2YoloV8-segment
将labelme数据标注格式转换为YoloV8语义分割数据集，并可自动划分训练集和验证集

![示例图片](https://github.com/KdaiP/labelme2YoloV8-segment/blob/main/demo.jpg)


## 配置

请先确保所有数据集图片和labelme标注文件都存放在一个文件夹内。

修改labelme_folder变量为该文件夹。例如：

```python
labelme_folder = "labelme_folder" 
```

## 使用

运行convert_folder.py

默认输出路径如下：

```python
output_folder = "output_dir" #存储yolo标注文件的文件夹
dataset_folder = "dataset" #存储划分好的数据集的文件夹
```

output_folder存放转换过后的YoloV8数据集图片和标签，dataset_folder存放划分完训练集和验证集的YoloV8数据集，并生成data.yaml文件。

如果不需要划分训练集和测试集，只需要注释掉这行：

```python
split_data(output_folder, dataset_folder)#划分训练集和验证级
```

## 训练

参照[YoloV8官方文档](https://docs.ultralytics.com/tasks/detect/)


示例：

```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  #加载预训练底模

# 训练
model.train(data='data.yaml', epochs=100, imgsz=640)
```

注：data.yaml文件在dataset_folder目录下面

## 疑难解答

* YOLOV8找不到训练集和测试集文件

检查convert_folder.py生成的data.yaml文件，修改其中的train、val和test路径

* with open(xxx,"w") as f 报错

在使用其他labelme转yolo数据集的程序可能会出现，原因是open只能创建文件而不能创建文件夹，需要事先创建好文件夹。

本程序用os.makedirs自动创建文件夹，一般情况下不会出现这个问题。