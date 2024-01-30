

# 人工智能第五次实验

杨帆-10215501416

此项目Github地址：

[shanse11/Multimodal-Sentiment-Analysis: AI-project5 (github.com)](https://github.com/shanse11/Multimodal-Sentiment-Analysis)

## 1 实验描述

### 摘要

​	人的情感是一种复杂且丰富的属性. 在现实中人物情感的获取来源有很多种途径, 比如一段微博评论, 一句音频录音, 还有人物的面部表情, 肢体动作等等. 在人于人或者机器与人的表达和交互过程中， 能够准确的把握相关方的态度可以极大的提升交互体验. 随着近几年基于深度学习的情感分析技术发展, 融合多个模态信息来预测情感状态成为一种趋势。

### 任务介绍

情感是人对客观事物所持的态度。最简单的情感可以分为积极（正向）、消极（负向）、中性，又称为情绪。 此外， 除了中性外， 更多样化的情感又可细分为：喜、怒、忧、悲、恐、惊等。 这些情感构成了人与人之间沟通交流的多样性，同时也包含了丰富的信息， 帮助我们掌握感兴趣对象在特定场景下的状态以及对相关事务的态度。 通过算法模型， 结合具体场景和数据分析特定对象的情感状态.

传统机器学习领域往往是针对单模态数据来进行处理和分析的，而真实世界中，一个事物的特征往往由多个模态共同组成，在许多场景下，我们需要共同考虑这些特征才能得到一个较好的表征。多模态学习正是在这样的背景下所提出的。传统的多模态学习模型往往由多个单模态模型及一系列融合技巧组成，而随着近年来预训练方法的提出，多模态预训练模型也正逐渐成为多模态理解模型的主流，尤其是在**视觉-语言领域**取得了显著的效果。

本次实验的主要任务为：对于一组匹配的图像文本对，要求给出一个情感倾向分类结果，使得该情感是对应图像文本对所表现出的情感倾向。

### 数据集介绍：

数据集中共包含了512*组匹配的图像-文本对。其中，训练集共有4000个样本，每个样本带有一个与之对应的情感分类标签。测试集共有511个样本，测试集的情感标签需要我们使用模型推理得到。

### 实验环境

编程语言：Python3.10  

编程工具：Pycharm

需要安装的Python库：pandas,opencv-python,numpy,scikit-learn,torch,transformers,Pillow

模型设计



## 2 多模态模型的设计：

多模态融合模型是一种能够同时处理多种类型输入数据（如图像、文本、音频等）的模型，通过整合和处理这些不同模态的信息，可以获得更全面、准确的信息表示和更好的性能。

在多模态融合模型的设计和实现中，通常包括以下几个主要步骤：

1. 特征提取：针对不同的模态数据，需要使用相应的技术来提取特征。例如，对于图像数据，可以使用卷积神经网络（如ResNet、VGG等）来提取图像特征；对于文本数据，可以使用预训练的语言模型（如BERT、GPT等）来提取文本特征。
2. 特征融合：在将不同模态的特征结合起来前，需要对它们的维度进行对齐或降维操作，使得它们在特征空间中具有相同的维度。然后，可以使用简单的拼接操作、加权融合、堆叠或注意力机制等方式将这些特征融合在一起，得到融合后的特征表示。
3. 训练和优化：多模态融合模型在许多任务上都显示出优越的表现，如图像标注、文本到图像生成、音频情感分析等。通过利用多个模态的互补信息，可以提取更细粒度、更全面的特征表示，从而增强模型的能力和性能。在训练过程中，通常使用监督学习的方式，通过最小化损失函数来优化模型参数。

对于我实现的情感分析分类的文本和图像双模态融合模型，该模型接受三种不同类型的输入：图像、文本和图像+文本。模型使用了一个图像特征提取器（ImageFeatureExtractor）和一个文本特征提取器（TextFeatureExtractor）来分别提取图像特征和文本特征。 模型的主要组成部分是三个分类器，它们分别用于处理图像+文本、仅文本和仅图像的情况。这些分类器都包括一系列的线性层和非线性激活函数，用于将提取到的特征映射到最终的输出类别。 前向传播函数根据输入类型的不同，选择合适的特征提取器和分类器来处理输入数据。如果输入类型为图像+文本，则图像和文本特征将被拼接起来，并通过第一个分类器进行处理。如果输入类型为仅文本，则只使用文本特征提取器和第三个分类器。如果输入类型为仅图像，则只使用图像特征提取器和第二个分类器。 最终，输出是模型对输入数据进行分类的结果。 总之，这个模型使用了多个特征提取器和分类器，以处理不同类型的输入数据，并生成相应的分类结果。通过融合不同模态的特征，模型可以更好地捕捉到输入数据的信息，提高分类性能。



## 3 实验步骤

### 数据准备：

text_process  函数使用了一个预训练的tokenizer，将输入的文本进行分词和编码，并将其转换成PyTorch张量。该函数接受一个文本列表和最大长度参数，并返回一个包含编码后的文本数据的列表。

Dataset类是一个自定义的数据集类，用于创建同时包含图像和文本输入的数据集。它接受图像路径、tokenized文本、标签和一个可选的图像转换函数作为参数。该类实现了len和getitem方法，分别返回数据集的长度和具体索引位置的数据样本。在getitem方法中，图像和文本数据通过调用图像转换函数进行处理，并包装成一个元组（image, input_ids, attention_mask, labels）返回。

### 模型训练：

循环迭代：根据设定的轮数(num_epochs)进行训练。

迭代训练：对训练数据进行迭代，将图像、文本编码、注意力掩码和标签传入模型进行训练。

网络更新：计算损失函数(loss)并进行反向传播，然后使用优化器(optimizer)更新网络参数。

ImageFeatureExtractor类是基于预训练的ResNet-50模型的图像特征提取器。在forward方法中，输入的图像通过ResNet-50模型，得到图像的特征表示。

TextFeatureExtractor类是基于预训练的BERT模型的文本特征提取器。在forward方法中，输入的文本通过BERT模型，得到文本的特征表示。其中，使用了BERT模型的第一个输出（outputs[1]）作为所有token的汇总表示。

MultiModalFusionModel继承自nn.Module，并在构造函数中初始化了图像特征提取器（image_feature）和文本特征提取器（text_feature），以及三个用于分类的线性层（classifier1、classifier2、classifier3）。根据参数input_type的取值，不同的输入类型会使用不同的分类器。具体地，如果input_type为1，则使用图像和文本的融合特征作为输入，并通过classifier1进行分类；如果input_type为2，则只使用文本特征进行分类，并通过classifier3进行分类；如果input_type不为1或2，则只使用图像特征进行分类，并通过classifier2进行分类。

### 辅助函数

get_valid_imagesPath_from_directory函数从给定的文件夹路径和DataFrame中提取有效的图像路径。对于DataFrame中的每个guid，它构建图像路径，并尝试使用cv2库读取图像。如果成功读取图像，将其路径添加到image_paths列表中。如果读取图像失败，将继续处理下一个guid。最后，返回所有有效图像路径的列表。

get_texts_from_textsPath函数从给定的文件夹路径和DataFrame中提取有效的文本数据。对于DataFrame中的每个guid，它构建文本文件的路径，并尝试使用open函数读取文本内容。如果成功读取文本文件，将其内容添加到texts列表中。如果文件不存在，将继续处理下一个guid。最后，返回所有有效文本数据的列表。

predict_model函数用于对测试集进行预测。在函数中，模型首先被设置为评估模式，然后对测试集中的图像和文本数据进行预测。预测结果根据预测概率的最大值确定标签，并将预测结果保存在predictions列表中，最后将其返回。

make_predictions函数用于对测试集进行标签预测。在函数中，模型首先被设置为评估模式（model.eval()），然后对测试集中的图像、文本数据进行预测。预测结果根据预测概率的最大值确定标签，并将预测结果保存在predictions列表中，最后将其返回。



由于篇幅原因不在这里展示代码部分了，在main.py中可以找到对应的函数。



## 4 实验结果

此实验一个epoch的运行时间很长，最长的一个epoch能运行近两个小时，因此没有时间来进行更多的参数调节来寻找最优解

选择批次batch size为64，学习率为5e-5,  epoch_number为6。

#### 多模态模型在验证集上的结果如下

```
start training use fusion model 
batch_size: 64, lr: 5e-05, Epoch 1/6, Train Loss: 1.0166, Train Acc: 0.5684, Val Acc: 0.5950
Total runtime: 2218.8862705230713 seconds
batch_size: 64, lr: 5e-05, Epoch 1/6, Train Loss: 1.0166, Train Acc: 0.5684, Val Acc: 0.5950
Total runtime: 2218.8862705230713 seconds
batch_size: 64, lr: 5e-05, Epoch 2/6, Train Loss: 0.9382, Train Acc: 0.5903, Val Acc: 0.6250
Total runtime: 2201.223906517029 seconds
batch_size: 64, lr: 5e-05, Epoch 3/6, Train Loss: 0.7886, Train Acc: 0.6800, Val Acc: 0.6713
Total runtime: 2188.1813452243805 seconds
batch_size: 64, lr: 5e-05, Epoch 4/6, Train Loss: 0.6385, Train Acc: 0.7478, Val Acc: 0.6475
Total runtime: 2189.3071982860565 seconds
batch_size: 64, lr: 5e-05, Epoch 5/6, Train Loss: 0.5528, Train Acc: 0.7913, Val Acc: 0.6637
Total runtime: 2179.8956129550934 seconds
batch_size: 64, lr: 5e-05, Epoch 6/6, Train Loss: 0.4865, Train Acc: 0.8194, Val Acc: 0.6175
Total runtime: 2189.676152229309 seconds
best_acc: 0.67125
```

最好的准确率达到了0.67125

#### 消融实验结果

只输入文本数据的训练结果：

```
start training only use text
batch_size: 64, lr: 5e-05, Epoch 1/6, Train Loss: 0.9329, Train Acc: 0.5822, Val Acc: 0.5887
Total runtime: 1956.312484741211 seconds
batch_size: 64, lr: 5e-05, Epoch 2/6, Train Loss: 0.8357, Train Acc: 0.6497, Val Acc: 0.6300
Total runtime: 1868.2162189483643 seconds
batch_size: 64, lr: 5e-05, Epoch 3/6, Train Loss: 0.7331, Train Acc: 0.7091, Val Acc: 0.6123
Total runtime: 1850.7900307178497 seconds
batch_size: 64, lr: 5e-05, Epoch 4/6, Train Loss: 0.6211, Train Acc: 0.7772, Val Acc: 0.6363
Total runtime: 1846.085061788559 seconds
batch_size: 64, lr: 5e-05, Epoch 5/6, Train Loss: 0.5289, Train Acc: 0.8250, Val Acc: 0.6450
Total runtime: 1848.3197782039642 seconds
batch_size: 64, lr: 5e-05, Epoch 6/6, Train Loss: 0.4590, Train Acc: 0.8528, Val Acc: 0.6512
Total runtime: 1844.0119452476501 seconds
best_acc: 0.68625

```

最好的准确率达到了0.6512

```
start training only use image
batch_size: 64, lr: 5e-05, Epoch 1/6, Train Loss: 1.0938, Train Acc: 0.5325, Val Acc: 0.5887
Total runtime: 78.86372208595276 seconds
batch_size: 64, lr: 5e-05, Epoch 2/6, Train Loss: 1.0082, Train Acc: 0.5903, Val Acc: 0.5887
Total runtime: 76.41352343559265 seconds
batch_size: 64, lr: 5e-05, Epoch 3/6, Train Loss: 0.9940, Train Acc: 0.5913, Val Acc: 0.5887
Total runtime: 77.61701679229736 seconds
batch_size: 64, lr: 5e-05, Epoch 4/6, Train Loss: 0.9656, Train Acc: 0.5931, Val Acc: 0.5900
Total runtime: 78.4812159538269 seconds
batch_size: 64, lr: 5e-05, Epoch 5/6, Train Loss: 0.9564, Train Acc: 0.5791, Val Acc: 0.5863
Total runtime: 79.01891136169434 seconds
batch_size: 64, lr: 5e-05, Epoch 6/6, Train Loss: 0.9231, Train Acc: 0.5825, Val Acc: 0.5875
Total runtime: 76.87196660041809 seconds
best_acc: 0.59

```

最好的准确率达到了0.59

可以发现只输入图像或只输入文本数据，模型依然可以学习到一定特征，验证集上的准确率随Epoch大体呈上升趋势，在训练集准确率较高、即过拟合时略微下降。但两者的验证集准确率均略低于融合模型，说明多模态融合模型相比单一模态输入的模型，能够将多种输入的特征结合起来，获得更全面、准确的特征表示，模型性能更强。

## 为什么会设计这样的模型？以及模型亮点。

我的模型结合了图像和文本的信息，利用预训练的ResNet-based图像特征提取器和预训练的BERT-based文本特征提取器。

亮点：

1. **多模态融合：**
   - 模型支持三种输入类型：
      - 图像和文本特征的融合（`input_type = 1`）。
      - 仅文本输入（`input_type = 2`）。
      - 仅图像输入（`input_type = 3`）。
   - `MultiModalFusionModel`根据输入类型使用不同的分类器来结合图像和文本的特征。
2. **文本和图像特征提取：**
   - 代码使用预训练的BERT模型进行文本特征提取。BERT是一种在文本数据中捕获上下文信息的强大模型。
   - 对于图像特征提取，代码利用了在ImageNet上预训练的ResNet50。使用预训练的卷积神经网络（CNN）如ResNet有助于捕获图像的层次特征。
4. **Dropout和非线性：**
   - 分类器中包含了Dropout层，以防止过拟合。
   - 在线性层之后使用了修正线性单元（ReLU）激活函数，引入非线性。
6. **使用Transformers库：**
   - 代码利用了Hugging Face的`transformers`库来获取BERT模型和分词器。
8. **预测：**
   - 在训练后，加载了模型以对测试集进行预测，并将结果保存在CSV文件中。

## 代码实现时遇到的bug以及解决方法

## 1 文本读取问题

开始使用gbk编码来读取文件，原因是因为文本文件涉及多语言，并且采用的编码方式也不一样，只用一种固定编码无法适应。然而尝试调用其他编码，依次解析不同文件，发现也会报错。例如当使用的是对应的字符编码（如GB2312）进行处理，由于文本中夹杂的部分特殊字符，因而仍会报错。于是最后通过选取与当前编码兼容但包含更多字符的编码GB18030去解码解决问题。

## 2 模型加载问题

最开始在加载模型时直接连接到huggingface，会超时然后显示失败，因为国内不能连接到huggingface，因此从huggingface的国内镜像网站下载了本地的bert-base-multilingual-cased模型。

