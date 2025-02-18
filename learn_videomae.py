import av
import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
import glob
import warnings
import random
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from av.video.frame import VideoFrame
from transformers import AutoImageProcessor, TimesformerForVideoClassification, VideoMAEImageProcessor, VideoMAEForVideoClassification, VivitImageProcessor, VivitForVideoClassification

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve

from dataloader_multimodal import natural_keys
from models import *

warnings.filterwarnings("ignore")

args = sys.argv
CKPT_NAME = args[1]

np.random.seed(0)
# model_name = "google/vivit-b-16x2-kinetics400"
model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
# model_name = "facebook/timesformer-base-finetuned-k400"
# model_name = "facebook/timesformer-hr-finetuned-k400"

image_processor = VideoMAEImageProcessor.from_pretrained(model_name)
print(f'processor: {model_name}', flush=True)


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def fix_image(pil_img):
    img = pil_img
    img = img.crop((80, 0, 560, 480))
    img = img.resize((350, 350))
    # img.save(f'./images/crop/crop_{random.randint(0, 10)}.png', quality=95)
    img_width, img_height = img.size
    img = img.crop((img_width/2-112, img_height/2-112, img_width/2+112, img_height/2+112))
    # img.save(f'./images/crop/crop_{random.randint(11, 20)}.png', quality=95)
    return img


def read_video_pyav(container, indices, frame_width=224):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    frames_tuple = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            # frame = frame.reformat(width=frame_width, height=frame_width)
            img = frame.to_image()
            normal_img = img.resize((int(img.width * (frame_width / img.height)), frame_width))
            normal_img = crop_center(img, frame_width, frame_width)
            normal_frame = VideoFrame.from_image(normal_img)
            fix_img = fix_image(img)
            fix_frame = VideoFrame.from_image(fix_img)
            # frame = VideoFrame.from_image(img)
            # frames.append(frame)
            frames_tuple.append((normal_frame, fix_frame))

    # return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    return np.stack([x[0].to_ndarray(format="rgb24") + x[1].to_ndarray(format="rgb24") for x in frames_tuple])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    # end_idx = np.random.randint(converted_len, seg_len)
    end_idx = seg_len
    # start_idx = end_idx - converted_len
    start_idx = 0
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, video_list):
        self.label_df = pd.read_csv('./data/annotation_result2.csv')
        valid_list = []
        label_list = []
        for i in range(len(video_list)):
            if self.label_df.iloc[i, 8]:
                valid_list.append(video_list[i])
                label = int(self.label_df.iloc[i, 9])
                label_list.append(label)

        self.video_list = valid_list
        self.label_list = label_list

    def __len__(self):
        '''動画の数を返す'''
        return len(self.video_list)

    def __getitem__(self, index):
        dir_path = self.video_list[index]
        label_id = self.label_list[index]

        container = av.open(dir_path)
        clip_len = 16
        indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=int(container.streams.video[0].frames / clip_len), seg_len=container.streams.video[0].frames)
        # print(f'frame={container.streams.video[0].frames}', flush=True)
        # print(f'frame_rate={int(container.streams.video[0].frames / 16)}', flush=True)
        # print(f'indices={indices}', flush=True)
        video = read_video_pyav(container, indices, frame_width=224)

        embeddings = image_processor(list(video), return_tensors="pt")
        embeddings = list(embeddings.data.values())[0]

        # with open('./log/data_split.log', 'a', encoding='utf-8') as f:
        #     f.write(f'{dir_path}\n')

        return embeddings, label_id


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        print(f'model: {model_name}', flush=True)
        model = VideoMAEForVideoClassification.from_pretrained(model_name)

        model.classifier = torch.nn.Linear(768, 768)

        self.main_model = model

        self.fc_middle = nn.Linear(in_features=768, out_features=512, bias=True)

        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x):

        bs = x.shape[0]
        out = x.view(bs, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
        out = self.main_model(out)

        out = out.logits
        out = self.fc_middle(out)
        out = self.fc_final(out)

        return out


def train(model, train_loader, val_loader, seed, n=10, device="cuda:0"):
    class_weight = [1.0, 2.86]
    weights = torch.tensor(class_weight).to(device)
    print(f'set weight loss: {class_weight}', flush=True)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    # criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.9)
    # optimizer = optim.SGD(params=model.parameters(), lr=0.0005, momentum=0.9)

    best_loss = 100
    early_stop = 5
    stop_count = 0

    print('--train start---', flush=True)
    for epoch in tqdm(range(n)):  # エポック数は適当
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        total = 0
        predict_all = []
        true_all = []

        model.train()
        for embeddings, label_id in train_loader:
            embeddings = embeddings.to(device)
            label_id = label_id.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            train_loss = criterion(outputs, label_id)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()

        train_loss_value = running_train_loss / len(train_loader)

        # print(f"Epoch {epoch + 1}, Train Loss: {train_loss_value}", flush=True)

        with torch.no_grad():
            model.eval()
            for embeddings, label_id in val_loader:
                embeddings = embeddings.to(device)
                label_id = label_id.to(device)
                outputs = model(embeddings)
                val_loss = criterion(outputs, label_id)

                _, predicted = torch.max(outputs, 1)
                running_val_loss += val_loss.item()
                total += label_id.size(0)
                running_accuracy += (predicted == label_id).sum().item()
                predict_all.extend(predicted.tolist())
                true_all.extend(label_id.tolist())

            val_loss_value = running_val_loss / len(val_loader)

            f1 = f1_score(true_all, predict_all)
            acc = accuracy_score(true_all, predict_all)

            print('\nCompleted epoch:', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Validation Loss is: %.4f' %val_loss_value, flush=True)
            print(f'Accuracy={acc}, F1={f1}', flush=True)
            # print(f'true = {true_all}\npred = {predict_all}', flush=True)

        if val_loss_value < best_loss:
            stop_count = 0
            # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
            torch.save(model.state_dict(), f'./checkpoints/videomae/{CKPT_NAME}-{seed}-best.pth')
            best_loss = val_loss_value
            print('save best model', flush=True)
        else:
            stop_count += 1
            if stop_count >= early_stop:
                torch.save(model.state_dict(), f'./checkpoints/videomae/{CKPT_NAME}-{seed}-last.pth')
                print('early stopping', flush=True)
                return
        if epoch == n-1:
            torch.save(model.state_dict(), f'./checkpoints/videomae/{CKPT_NAME}-{seed}-last.pth')


def finetune(model, train_loader, val_loader, seed, n=10, device="cuda:0"):
    # weights = torch.tensor([1.0, 3.0]).to(device)
    # print('set weight loss', flush=True)
    # criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    model_state = model.state_dict()
    pretrain_state = torch.load('./checkpoints/mae-pretrain/mae-pretrain-6-e29.pth')
    state = {k:v for k, v in pretrain_state.items() if k in model_state and v.size() == model_state[k].size()}
    
    load_state = {}
    for k, v in model_state.items():
        pretrain_k = k.replace('main_model.', '')
        if pretrain_k in pretrain_state and v.size() == pretrain_state[pretrain_k].size():
            print(f'load weight of {k}', flush=True)
            load_state[k] = pretrain_state[pretrain_k]
        else:
            load_state[k] = v

    # print(load_state.keys(), flush=True)
    model.load_state_dict(load_state)
    # model_state.update(state)
    # model.load_state_dict(model_state)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.9)

    best_loss = 100
    early_stop = 5
    stop_count = 0

    print('--train start---', flush=True)
    for epoch in tqdm(range(n)):  # エポック数は適当
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        total = 0
        predict_all = []
        true_all = []

        model.train()
        for embeddings, label_id in train_loader:
            embeddings = embeddings.to(device)
            label_id = label_id.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            train_loss = criterion(outputs, label_id)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()

        train_loss_value = running_train_loss / len(train_loader)

        # print(f"Epoch {epoch + 1}, Train Loss: {train_loss_value}", flush=True)

        with torch.no_grad():
            model.eval()
            for embeddings, label_id in val_loader:
                embeddings = embeddings.to(device)
                label_id = label_id.to(device)
                outputs = model(embeddings)
                val_loss = criterion(outputs, label_id)

                _, predicted = torch.max(outputs, 1)
                running_val_loss += val_loss.item()
                total += label_id.size(0)
                running_accuracy += (predicted == label_id).sum().item()
                predict_all.extend(predicted.tolist())
                true_all.extend(label_id.tolist())

            val_loss_value = running_val_loss / len(val_loader)

            f1 = f1_score(true_all, predict_all)
            acc = accuracy_score(true_all, predict_all)

            print('\nCompleted epoch:', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Validation Loss is: %.4f' %val_loss_value, flush=True)
            print(f'Accuracy={acc}, F1={f1}', flush=True)
            # print(f'true = {true_all}\npred = {predict_all}', flush=True)

        if val_loss_value < best_loss:
            stop_count = 0
            # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
            torch.save(model.state_dict(), f'./checkpoints/videomae/{CKPT_NAME}-{seed}-best.pth')
            best_loss = val_loss_value
            print('save best model', flush=True)
        else:
            stop_count += 1
            if stop_count >= early_stop:
                torch.save(model.state_dict(), f'./checkpoints/videomae/{CKPT_NAME}-{seed}-last.pth')
                print('early stopping', flush=True)
                return
        if epoch == n-1:
            torch.save(model.state_dict(), f'./checkpoints/videomae/{CKPT_NAME}-{seed}-last.pth')


def test(model, test_loader, seed, device="cuda:0"):
    # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
    model.load_state_dict(torch.load( f'./checkpoints/videomae/{CKPT_NAME}-{seed}-best.pth'))

    running_accuracy = 0.0
    total = 0
    true_all = []
    predict_all = []
    
    preds_all = []

    path_list = []

    print('--- test start ---', flush=True)

    model.eval()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)

        for embeddings, label_id in test_loader:
            embeddings = embeddings.to(device)
            label_id = label_id.to(device)
            outputs = model(embeddings)

            _, predicted = torch.max(outputs, 1)
            total += label_id.size(0)
            running_accuracy += (predicted == label_id).sum().item()

            true_all.extend(label_id.tolist())
            predict_all.extend(predicted.tolist())

            preds = softmax(outputs)[:, 1]
            preds_all.extend(preds.tolist())

            # path_list.extend(dir_path)

        print(f'true = {true_all}', flush=True)
        print(f'predict = {predict_all}', flush=True)

        f1 = f1_score(true_all, predict_all)
        pre = precision_score(true_all, predict_all)
        recall = recall_score(true_all, predict_all)
        acc = accuracy_score(true_all, predict_all)
        print(f'Accuracy: {acc}\nPrecision: {pre}\nRecall: {recall}\nF1: {f1}', flush=True)
        # print(f'len: {len(predict_all)}, anomaly: {predict_all.count(1)}', flush=True)
        # print(f'softmax ave: {sum(preds_all) / len(preds_all)}', flush=True)

        # precision, recall, thresholds = precision_recall_curve(true_all, preds_all)
        # np.save(f'./analysis/mae/p-{seed}.npy', precision)
        # np.save(f'./analysis/mae/r-{seed}.npy', recall)
        # np.save(f'./analysis/mae/th-{seed}.npy', thresholds)

        # with open(f'./score/mae/logits.txt', 'a', encoding='utf-8') as f:
        #     f.write(str(f1))
        #     f.write('\n')


def main(seed_list=[42]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    video_list = glob.glob('./data/video/*.mp4')
    video_list = sorted(video_list, key=natural_keys)

    for seed in seed_list:
        print(f'\n\n--- SEED={seed} ---', flush=True)

        all_dataset = MyDataset(video_list)

        all_len = len(all_dataset)
        train_len = int(0.7 * all_len)
        val_len = int((all_len-train_len) / 2)
        test_len = all_len - train_len - val_len

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset=all_dataset, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

        print(f'all: {len(all_dataset)}, train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}', flush=True)

        model = VideoMAEModel()

        model = model.to(device)

        print('model loaded.', flush=True)

        # 学習
        train(model=model, train_loader=train_loader, val_loader=val_loader, seed=seed, n=20, device=device)

        # 評価
        test(model=model, test_loader=test_loader, seed=seed, device=device)


if __name__ == '__main__':
    seed_list = [25, 96, 13]
    print(CKPT_NAME, flush=True)
    main(seed_list=seed_list)