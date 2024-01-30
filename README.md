# 多模态情感分析

该仓库存储了使用图片+文本构建多模态模型分析情感的代码。

## 设置

你可以通过运行以下代码安装本项目所需依赖。

```
pip install -r requirements.txt
```



## 仓库结构

以下是一些重要文件及其描述。

```
|-- data # 图片和文本数据
|-- README.md
|-- Report.md # 实验报告
|-- main.py # 实现代码
|-- predict.txt # 预测输出文件
|-- requirement.txt # 运行所需依赖
|-- test_with_label.txt # 需要预测的文件
|-- train.txt # 训练数据
```



## 代码运行的流程

下载所有文件，由于我的离线下载模型太大无法上传github，因此上传至了网盘，

链接：https://pan.baidu.com/s/1sdJzZ0KXJqcPZm28vS0_xA 
提取码：1234

在下载好所有文件后在此文件目录下终端运行

1在命令行中输入以下代码，即可运行图像和文本的融合特征

```
 python main.py --input_type 1
```

2在命令行中输入以下代码，即可运行只输入文本的消融实验

```
 python main.py --input_type 2
```

3在命令行中输入以下代码，即可运行只输入图像的消融实验

```
 python main.py --input_type 3
```

### 参考库

本项目并未参考其他库