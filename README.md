# labelme2YoloV8-segment
将labelme数据标注格式转换为YoloV8语义分割数据集，并可自动划分训练集和验证集

![示例图片](https://github.com/KdaiP/labelme2YoloV8-segment/blob/main/demo.jpg)


## 使用

请先确保所有数据集图片和labelme标注文件都存放在一个文件夹内。脚本根据文件名对图片-标注进行匹配。

```shell
python convert_folder.py 待转换的文件夹 输出文件夹 训练集占比
```

示例：

```shell
python convert_folder.py examples datasets 0.8
```


## 训练

参照[YoloV8官方文档](https://docs.ultralytics.com/tasks/segment/)


示例：

```python
from ultralytics import YOLO
from ultralytics import settings

settings.update({'datasets_dir': './'})
model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

if __name__ == '__main__':
    # Train the model
    results = model.train(data='./datasets/yolo.yaml', epochs=100, imgsz=640)
```


## 疑难解答

* YOLOV8找不到训练集和测试集文件

YOLOV8在查找路径时，会将三个路径拼接到一起：
setting中的datasets_dir
数据集yaml中的path
数据集yaml中的train、value

可以通过以下方式来修改 datasets_dir：
```python
from ultralytics import settings
settings.update({'datasets_dir': './'})
```
