import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision.models import resnet50
import argparse
import time


# 分词,并限制文本最大长度
def text_process(texts, max_length):
    tokenized_texts = [
        tokenizer(text, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt") for text in
        texts]
    return tokenized_texts


# 创建同时包含图像和文本的数据集的函数
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, tokenized_texts, labels, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.input_ids = [x['input_ids'] for x in tokenized_texts]
        self.attention_mask = [x['attention_mask'] for x in tokenized_texts]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index].clone().detach()
        attention_mask = self.attention_mask[index].clone().detach()
        labels = torch.tensor(self.labels[index])
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, input_ids, attention_mask, labels


class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.resnet = resnet50(pretrained=True)

    def forward(self, image):
        features = self.resnet(image)
        return features


class TextFeatureExtractor(nn.Module):
    def __init__(self):
        super(TextFeatureExtractor, self).__init__()
        self.bert = pretrained_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        output = pooled_output
        return output


# 多模态融合模型
class MultiModalFusionModel(nn.Module):
    def __init__(self, num_classes, input_type):
        super(MultiModalFusionModel, self).__init__()
        self.image_feature = ImageFeatureExtractor()
        self.text_feature = TextFeatureExtractor()
        self.input_type = input_type
        self.classifier1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1768, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, image, input_ids, attention_mask):
        if self.input_type == 1:
            image_features = self.image_feature(image)
            text_features = self.text_feature(input_ids, attention_mask)
            fusion_features = torch.cat((text_features, image_features), dim=-1)
            output = self.classifier1(fusion_features)
        # 仅文本
        elif self.input_type == 2:
            text_features = self.text_feature(input_ids, attention_mask)
            output = self.classifier3(text_features)
        # 仅图像
        else:
            image_features = self.image_feature(image)
            output = image_features
            output = self.classifier2(image_features)
        return output


# 标签预测函数
def make_predictions(model, test_loader, device):
    model.eval()
    predictions = []
    for images, input_ids, attention_mask, _ in test_loader:
        images = images.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
    return predictions


def get_valid_imagesPath_from_directory(folder_path, df):
    image_paths = []
    for ind in df['guid']:
        image_path = folder_path + str(ind) + ".jpg"
        try:
            image = cv2.imread(image_path)
            image_paths.append(image_path)
        except Exception as e:
            continue

    return image_paths


def get_texts_from_textsPath(folder_path, df):
    texts = []
    for ind in df['guid']:
        file = folder_path + str(ind) + ".txt"
        try:
            with open(file, "r", encoding="GB18030",errors='ignore') as infile:
                content = infile.read()
                texts.append(content)
        except FileNotFoundError:
            continue
    return texts

def predict_model(model, test_loader, device):
    model.eval()
    predictions = []
    for images,input_ids, attention_mask,  _ in test_loader:
        images = images.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(images, input_ids,attention_mask)
            _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
    return predictions


if __name__ == "__main__":
    path = "./bert-base-multilingual-cased/"
    pretrained_model = BertModel.from_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained(path)
    transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor(), ])

    # 数据读取与预处理
    folder_path = './data/'
    train_label = pd.read_csv('train.txt', sep=",")
    column_dict = {"positive": 0, "negative": 1, "neutral": 2}
    new_train_label = train_label.replace({"tag": column_dict})
    labels = list(new_train_label['tag'])
    image_paths = []
    for seq_num in new_train_label['guid']:
        image_path = folder_path + str(seq_num) + '.jpg'
        image = cv2.imread(image_path)
        image_paths.append(image_path)
    texts = []
    for seq_num in new_train_label['guid']:
        path = folder_path + str(seq_num) + '.txt'
        with open(path, "r", encoding='gb18030') as file:
            content = file.read()
            texts.append(content)
    image_paths_train, image_paths_valid, texts_train, texts_valid, labels_train, labels_valid = train_test_split(
        image_paths, texts, labels, test_size=0.2, random_state=10)
    max_length = 131
    # 文本预处理
    tokenized_texts_train = text_process(texts_train, max_length)
    tokenized_texts_valid = text_process(texts_valid, max_length)

    dataset_train = Dataset(image_paths_train, tokenized_texts_train, labels_train, transform)
    dataset_valid = Dataset(image_paths_valid, tokenized_texts_valid, labels_valid, transform)

    # 训练
    # 参数
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    lr = 5e-5
    num_epochs = 6
    parser = argparse.ArgumentParser()
    # type选1代表既输入文本也输入图像，2代表仅输入文本，3代表仅输入图像
    parser.add_argument('--input_type', type=int, default=1, help='0-only image 1-only text 2-fusion')
    args = parser.parse_args()
    input_type = 1
    input_type = args.input_type
    if input_type == 1:
        print("start training use fusion model")
    if input_type == 2:
        print("start training only use text")
    if input_type == 3:
        print("start training only use image")
    model = MultiModalFusionModel(num_classes, input_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    best_model = None

    # 数据加载
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型训练
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        train_correct = 0
        for images, input_ids, attention_mask, labels in loader_train:
            images = images.to(device)
            input_ids = input_ids.squeeze(1).to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        tr_loss = train_loss / len(loader_train)
        tr_acc = train_correct.item() / len(loader_train.dataset)

        predictions = make_predictions(model, loader_valid, device)

        val_predictions = np.array(predictions)
        val_labels = np.array(labels_valid)
        val_acc = (val_predictions == val_labels).sum() / len(val_labels)
        if (val_acc > best_acc):
            best_acc = val_acc
            best_model = model
            torch.save(model, 'best_model.pt')
        end_time = time.time()
        total_time = end_time - start_time
        print(
            f"batch_size: {batch_size}, lr: {lr}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Total runtime: {total_time} seconds")
    print(f"best_acc: {best_acc}")
    print("training finished")
    print("start predicting...")
    # 读取test文件
    test_path = "test_without_label.txt"
    test_df = pd.read_csv(test_path, sep=",")
    test_df.iloc[:, -1] = 0
    test_labels = np.array(test_df['tag'])
    # tests数据处理并构建数据加载器
    image_paths_test = get_valid_imagesPath_from_directory(folder_path, test_df)
    test_texts = get_texts_from_textsPath(folder_path, test_df)
    tokenized_texts_test = text_process(test_texts)
    dataset_test = Dataset(image_paths_test, tokenized_texts_test, test_labels, transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    best_model = torch.load('best_model.pt.pt').to(device)
    test_predictions = predict_model(best_model, loader_test, device)
    test_predictions = np.array(test_predictions)
    column_dict_ = {0: "positive", 1: "negative", 2: "neutral"}
    test_df['tag'] = test_predictions
    pre_df = test_df.replace({"tag": column_dict_})
    pre_df.to_csv('predict.txt', sep=',', index=False)
    print("prediction finished")
