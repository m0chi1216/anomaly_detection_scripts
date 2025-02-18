import av
import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
import glob
import warnings
import sys

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from av.video.frame import VideoFrame
from transformers import AutoImageProcessor, TimesformerForVideoClassification, VideoMAEImageProcessor, VideoMAEForVideoClassification

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.inference import InferenceModel

from dataloader_multimodal import natural_keys

from sklearn.neighbors import NearestNeighbors

from models import *
import pandas as pd

warnings.filterwarnings("ignore")


np.random.seed(0)

# model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
# model_name = "MCG-NJU/videomae-base-finetuned-ssv2"
model_name = "MCG-NJU/videomae-huge-finetuned-kinetics"

image_processor = VideoMAEImageProcessor.from_pretrained(model_name)
print(f'processor: {model_name}', flush=True)

args = sys.argv
CKPT_NAME = args[1]


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


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
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            # frame = frame.reformat(width=frame_width, height=frame_width)
            img = frame.to_image()
            img = img.resize((int(img.width * (frame_width / img.height)), frame_width))
            img = crop_center(img, frame_width, frame_width)
            frame = VideoFrame.from_image(img)
            frames.append(frame)

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


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

        self.fc_final = nn.Linear(in_features=768, out_features=512, bias=True)

    def forward(self, x):

        bs = x.shape[0]
        out = x.view(bs, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
        out = self.main_model(out)

        out = out.logits
        out = self.fc_final(out)

        return out


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def train(model, loss_func, train_loader, val_loader, seed, n=10, device="cuda:0"):
    # optimizer = optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.9)
    # loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.001)
    # optimizer = optim.SGD([
    #         {'params': model.parameters()},
    #         {'params': loss_func.parameters(), 'lr': 0.001}
    #     ], lr=0.0001, momentum=0.9)
    optimizer = optim.SGD([
            {'params': model.parameters()},
            {'params': loss_func.parameters(), 'lr': 0.001}
        ], lr=0.0001, momentum=0.9)

    best_loss = 100
    early_stop = 3
    stop_count = 0

    print('--train start---', flush=True)
    for epoch in tqdm(range(n)):  # エポック数は適当
        running_train_loss = 0.0
        running_val_loss = 0.0

        model.train()
        for embeddings, label_id in train_loader:
            embeddings = embeddings.to(device)
            label_id = label_id.to(device)
            optimizer.zero_grad()
            # loss_optimizer.zero_grad()
            outputs = model(embeddings)
            train_loss = loss_func(outputs, label_id)
            train_loss.backward()
            optimizer.step()
            # loss_optimizer.step()
            running_train_loss += train_loss.item()

        train_loss_value = running_train_loss / len(train_loader)

        # print(f"Epoch {epoch + 1}, Train Loss: {train_loss_value}", flush=True)

        with torch.no_grad():
            model.eval()
            for embeddings, label_id in val_loader:
                embeddings = embeddings.to(device)
                label_id = label_id.to(device)
                outputs = model(embeddings)
                val_loss = loss_func(outputs, label_id)
                running_val_loss += val_loss.item()

            val_loss_value = running_val_loss / len(val_loader)

            print('Completed epoch:', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Validation Loss is: %.4f' %val_loss_value, flush=True)


        if val_loss_value < best_loss:
            stop_count = 0
            # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
            torch.save(model.state_dict(), f'./checkpoints/mae-metric/{CKPT_NAME}-{seed}-best.pth')
            best_loss = val_loss_value
            print('save best model', flush=True)
        else:
            stop_count += 1
            if stop_count >= early_stop:
                torch.save(model.state_dict(), f'./checkpoints/mae-metric/{CKPT_NAME}-{seed}-last.pth')
                print('early stopping', flush=True)
                return
        if epoch == n-1:
            torch.save(model.state_dict(), f'./checkpoints/mae-metric/{CKPT_NAME}-{seed}-last.pth')


def test(model, test_loader, seed, device="cuda:0"):
    # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
    model.load_state_dict(torch.load( f'./checkpoints/mae-metric/{CKPT_NAME}-{seed}-best.pth'))

    _predicted_metrics = []
    _true_labels = []

    print('--- test start ---', flush=True)

    model.eval()
    with torch.no_grad():
        for i, (inputs,  labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            metric = model(inputs).detach().cpu().numpy()
            metric = metric.reshape(metric.shape[0], metric.shape[1])
            _predicted_metrics.append(metric)
            _true_labels.append(labels.detach().cpu().numpy())
    
    predicted_metrics = np.concatenate(_predicted_metrics)
    true_labels = np.concatenate(_true_labels)
    print(predicted_metrics.shape, true_labels.shape)
    np.save(f'./metrics/mae/{CKPT_NAME}-{seed}.npy', predicted_metrics)
    np.save(f'./metrics/mae/labels-{CKPT_NAME}-{seed}.npy', true_labels)


def infer(model, train_dataset, test_dataset, seed, device):
    # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
    model.load_state_dict(torch.load( f'./checkpoints/mae-metric/{CKPT_NAME}-{seed}-best.pth'))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    print('--- infer start ---', flush=True)

    train_embeddings = []
    train_labels = []
    test_embeddings = []
    test_labels = []
    
    model.eval()
    with torch.no_grad():
        for embeddings, label_id in train_loader:
            embeddings, label_id = embeddings.to(device), label_id.to(device)
            metric = model(embeddings).detach().cpu().numpy()
            metric = metric.reshape(metric.shape[0], metric.shape[1])
            train_embeddings.append(metric)
            train_labels.extend(label_id.tolist())
        train_embeddings = np.concatenate(train_embeddings)

        for embeddings, label_id in test_loader:
            embeddings, label_id = embeddings.to(device), label_id.to(device)
            metric = model(embeddings).detach().cpu().numpy()
            metric = metric.reshape(metric.shape[0], metric.shape[1])
            test_embeddings.append(metric)
            test_labels.extend(label_id.tolist())
        test_embeddings = np.concatenate(test_embeddings)

    print('-- get embedding -- ', flush=True)

    k_list = [1, 10, 30, 50]
    # weight = 2.86
    weight = 1.0
    for k in k_list:
        print(f'--- k={k} ---', flush=True)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train_embeddings)

        distances, indices = nbrs.kneighbors(test_embeddings)

        pred = []
        true_labels = []

        for i in range(len(indices)):
            true_label = int(test_labels[i])
            labels = [int(train_labels[idx]) for idx in indices[i]]
            if i == 0:
                print(true_label, flush=True)
                print(labels, flush=True)
            if labels.count(0) > labels.count(1) * weight:
                pred_label = 0
            else:
                pred_label = 1
            pred.append(pred_label)
            true_labels.append(true_label)

        f1 = f1_score(true_labels, pred)
        pre = precision_score(true_labels, pred)
        recall = recall_score(true_labels, pred)
        acc = accuracy_score(true_labels, pred)
        print(f'Accuracy: {acc}\nPrecision: {pre}\nRecall: {recall}\nF1: {f1}\n', flush=True)

        # with open(f'./score/mae/{k}.txt', 'a', encoding='utf-8') as f:
        #     f.write(str(f1))
        #     f.write('\n')

    print(f'true = {true_labels}', flush=True)
    print(f'pred = {pred}', flush=True)


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

        print(f'all: {len(all_dataset)} -> train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}', flush=True)

        model = VideoMAEFeatureModel()

        # for name, param in model.named_parameters():
        #     if "classifier" in name or "fc_final" in name:
        #         param.requires_grad = True
        #         params_to_update.append(param)
        #         print(name, flush=True)
        #     else:
        #         param.requires_grad = False

        model = model.to(device)

        print('model loaded.', flush=True)

        # loss_func = losses.ArcFaceLoss(num_classes=2, embedding_size=512, margin=28.6, scale=64)
        loss_func = losses.RankedListLoss(margin=35, Tn=0, imbalance=0.75, alpha=None, Tp=0)

        # 学習
        train(model=model, loss_func=loss_func, train_loader=train_loader, val_loader=val_loader, seed=seed, n=20, device=device)

        # 評価
        test(model=model, test_loader=test_loader, seed=seed, device=device)

        # 推論
        infer(model=model, train_dataset=train_dataset, test_dataset=test_dataset, seed=seed, device=device)


if __name__ == '__main__':
    seed_list = [25, 96, 13]
    print(CKPT_NAME, flush=True)
    main(seed_list=seed_list)