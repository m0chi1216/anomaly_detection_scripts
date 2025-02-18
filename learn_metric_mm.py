import av
import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
import glob
import warnings
import sys
import random
import pandas as pd
import pickle

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from av.video.frame import VideoFrame
from transformers import AutoImageProcessor, TimesformerForVideoClassification, VideoMAEImageProcessor, VideoMAEForVideoClassification, VivitImageProcessor, VivitForVideoClassification

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dataloader_multimodal import *
from models import CMAModel, DecisionModel, CMAFeatureModel

from sklearn.neighbors import NearestNeighbors

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


def train(model, loss_func, train_loader, val_loader, seed, n=10, device="cuda:0"):
    # optimizer = optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.9)
    # loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.0001)
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
        running_accuracy = 0.0
        running_val_loss = 0.0
        total = 0
        predict_all = []
        true_all = []

        model.train()
        for video_embeddings, audio_embeddings, label_id in train_loader:
            video_embeddings, audio_embeddings, label_id = video_embeddings.to(device), audio_embeddings.to(device), label_id.to(device)
            optimizer.zero_grad()
            # loss_optimizer.zero_grad()
            outputs = model(video_embeddings, audio_embeddings)
            train_loss = loss_func(outputs, label_id)
            train_loss.backward()
            optimizer.step()
            # loss_optimizer.step()
            running_train_loss += train_loss.item()

        train_loss_value = running_train_loss / len(train_loader)

        # print(f"Epoch {epoch + 1}, Train Loss: {train_loss_value}", flush=True)

        with torch.no_grad():
            model.eval()
            for video_embeddings, audio_embeddings, label_id  in val_loader:
                video_embeddings, audio_embeddings, label_id = video_embeddings.to(device), audio_embeddings.to(device), label_id.to(device)
                outputs = model(video_embeddings, audio_embeddings)
                val_loss = loss_func(outputs, label_id)
                running_val_loss += val_loss.item()

            val_loss_value = running_val_loss / len(val_loader)

            print('\nCompleted epoch:', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Validation Loss is: %.4f' %val_loss_value, flush=True)

        if val_loss_value < best_loss:
            stop_count = 0
            # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
            torch.save(model.state_dict(), f'./checkpoints/mm-metric-freeze/{CKPT_NAME}-{seed}-best.pth')
            best_loss = val_loss_value
            print('save best model', flush=True)
        else:
            stop_count += 1
            if stop_count >= early_stop:
                torch.save(model.state_dict(), f'./checkpoints/mm-metric-freeze/{CKPT_NAME}-{seed}-last.pth')
                print('early stopping', flush=True)
                break
        if epoch == n-1:
            torch.save(model.state_dict(), f'./checkpoints/mm-metric-freeze/{CKPT_NAME}-{seed}-last.pth')
    return best_loss


def train_all(model, loss_func, train_loader, val_loader, seed, n=10, device="cuda:0", best_loss=100):
    # optimizer = optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.9)
    # loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.0001)
    # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
    model.load_state_dict(torch.load( f'./checkpoints/mm-metric-freeze/{CKPT_NAME}-{seed}-best.pth'))
    torch.save(model.state_dict(), f'./checkpoints/mm-metric/{CKPT_NAME}-{seed}-best.pth')

    optimizer = optim.SGD([
            {'params': model.parameters()},
            {'params': loss_func.parameters(), 'lr': 0.001}
        ], lr=0.0001, momentum=0.9)

    early_stop = 3
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
        for video_embeddings, audio_embeddings, label_id in train_loader:
            video_embeddings, audio_embeddings, label_id = video_embeddings.to(device), audio_embeddings.to(device), label_id.to(device)
            optimizer.zero_grad()
            # loss_optimizer.zero_grad()
            outputs = model(video_embeddings, audio_embeddings)
            train_loss = loss_func(outputs, label_id)
            train_loss.backward()
            optimizer.step()
            # loss_optimizer.step()
            running_train_loss += train_loss.item()

        train_loss_value = running_train_loss / len(train_loader)

        # print(f"Epoch {epoch + 1}, Train Loss: {train_loss_value}", flush=True)

        with torch.no_grad():
            model.eval()
            for video_embeddings, audio_embeddings, label_id  in val_loader:
                video_embeddings, audio_embeddings, label_id = video_embeddings.to(device), audio_embeddings.to(device), label_id.to(device)
                outputs = model(video_embeddings, audio_embeddings)
                val_loss = loss_func(outputs, label_id)
                running_val_loss += val_loss.item()

            val_loss_value = running_val_loss / len(val_loader)

            print('\nCompleted epoch:', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Validation Loss is: %.4f' %val_loss_value, flush=True)

        if val_loss_value < best_loss:
            stop_count = 0
            # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
            torch.save(model.state_dict(), f'./checkpoints/mm-metric/{CKPT_NAME}-{seed}-best.pth')
            best_loss = val_loss_value
            print('save best model', flush=True)
        else:
            stop_count += 1
            if stop_count >= early_stop:
                torch.save(model.state_dict(), f'./checkpoints/mm-metric/{CKPT_NAME}-{seed}-last.pth')
                print('early stopping', flush=True)
                return
        if epoch == n-1:
            torch.save(model.state_dict(), f'./checkpoints/mm-metric/{CKPT_NAME}-{seed}-last.pth')
            


def test(model, test_loader, seed, device="cuda:0"):
    # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
    model.load_state_dict(torch.load( f'./checkpoints/mm-metric/{CKPT_NAME}-{seed}-best.pth'))

    _predicted_metrics = []
    _true_labels = []

    print('--- test start ---', flush=True)

    model.eval()
    with torch.no_grad():
        for v_emb, a_emb, labels in test_loader:
            v_emb, a_emb, labels = v_emb.to(device), a_emb.to(device), labels.to(device)
            metric = model(v_emb, a_emb).detach().cpu().numpy()
            metric = metric.reshape(metric.shape[0], metric.shape[1])
            _predicted_metrics.append(metric)
            _true_labels.append(labels.detach().cpu().numpy())
    
    predicted_metrics = np.concatenate(_predicted_metrics)
    true_labels = np.concatenate(_true_labels)
    print(predicted_metrics.shape, true_labels.shape)
    np.save(f'./metrics/mm/{CKPT_NAME}-{seed}.npy', predicted_metrics)
    np.save(f'./metrics/mm/labels-{CKPT_NAME}-{seed}.npy', predicted_metrics)


def infer(model, train_dataset, test_dataset, seed, device="cuda:0"):
    print(f'load checkpoint: ./checkpoints/mm-metric/{CKPT_NAME}-{seed}-best.pth', flush=True)
    model.load_state_dict(torch.load( f'./checkpoints/mm-metric/{CKPT_NAME}-{seed}-best.pth'))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    print(f'train size: {len(train_dataset)}, test size: {len(test_dataset)}', flush=True)

    print('--- infer start ---', flush=True)

    train_embeddings = []
    train_labels = []
    test_embeddings = []
    test_labels = []
    file_list = []
    
    model.eval()
    with torch.no_grad():
        # for video_embeddings, audio_embeddings, label_id in train_loader:
        #     video_embeddings, audio_embeddings, label_id = video_embeddings.to(device), audio_embeddings.to(device), label_id.to(device)
        #     metric = model(video_embeddings, audio_embeddings).detach().cpu().numpy()
        #     metric = metric.reshape(metric.shape[0], metric.shape[1])
        #     train_embeddings.append(metric)
        #     train_labels.extend(label_id.tolist())
        # train_embeddings = np.concatenate(train_embeddings)

        # np.save(f'./metrics/mm/{CKPT_NAME}-train-{seed}.npy', train_embeddings)

        # train_labels = np.array(train_labels)
        # np.save(f'./metrics/mm/{CKPT_NAME}-trainlabel-{seed}.npy', train_labels)
        train_embeddings = np.load(f'./metrics/mm/{CKPT_NAME}-train-{seed}.npy')
        train_labels = np.load(f'./metrics/mm/{CKPT_NAME}-trainlabel-{seed}.npy')

        for video_embeddings, audio_embeddings, label_id in test_loader:
            video_embeddings, audio_embeddings, label_id = video_embeddings.to(device), audio_embeddings.to(device), label_id.to(device)
            metric = model(video_embeddings, audio_embeddings).detach().cpu().numpy()
            metric = metric.reshape(metric.shape[0], metric.shape[1])
            test_embeddings.append(metric)
            test_labels.extend(label_id.tolist())
            file_list.extend(file_name)
        test_embeddings = np.concatenate(test_embeddings)

    print('-- get embedding -- ', flush=True)

    k_list = [10]
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


def create_emb(model, train_dataset, seed, device="cuda:0"):
    print(f'load checkpoint: ./checkpoints/mm-metric/{CKPT_NAME}-{seed}-best.pth', flush=True)
    model.load_state_dict(torch.load( f'./checkpoints/mm-metric/{CKPT_NAME}-{seed}-best.pth'))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    print(f'train size: {len(train_dataset)}', flush=True)

    print('--- creation start ---', flush=True)

    train_embeddings = []
    train_labels = []
    reasons = []
    
    model.eval()
    with torch.no_grad():
        for video_embeddings, audio_embeddings, label_id, reason in train_loader:
            video_embeddings, audio_embeddings = video_embeddings.to(device), audio_embeddings.to(device)
            metric = model(video_embeddings, audio_embeddings).detach().cpu().numpy()
            metric = metric.reshape(metric.shape[0], metric.shape[1])
            train_embeddings.append(metric)
            train_labels.extend(label_id.tolist())
            reasons.extend(reason)
        train_embeddings = np.concatenate(train_embeddings)

        np.save(f'./metrics/mm/{CKPT_NAME}-embedding-{seed}.npy', train_embeddings)

        train_labels = np.array(train_labels)
        np.save(f'./metrics/mm/{CKPT_NAME}-label-{seed}.npy', train_labels)

        f = open(f'./metrics/mm/{CKPT_NAME}-reason-{seed}.txt', 'wb')
        pickle.dump(reasons, f)
    
    print('--- finish ---')


def main(seed_list=[42]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    video_list = glob.glob('./data/video/*.mp4')
    video_list = sorted(video_list, key=natural_keys)
    audio_list = glob.glob('./data/audio/*.wav')
    audio_list = sorted(audio_list, key=natural_keys)

    for seed in seed_list:
        print(f'\n\n--- SEED={seed} ---', flush=True)

        all_dataset = VideoDataset(video_list, audio_list)

        all_len = len(all_dataset)
        train_len = int(0.7 * all_len)
        val_len = int((all_len-train_len) / 2)
        test_len = all_len - train_len - val_len

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset=all_dataset, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

        print(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}', flush=True)


        model = CMAFeatureModel(seed=seed, is_freeze=True)

        model = model.to(device)

        loss_func = losses.RankedListLoss(margin=35, Tn=0, imbalance=0.75, alpha=None, Tp=0)

        # 学習
        best_loss = train(model=model, loss_func=loss_func, train_loader=train_loader, val_loader=val_loader, seed=seed, n=20, device=device)

        # 全学習
        model = CMAFeatureModel(seed=seed, is_freeze=False)
        model = model.to(device)
        train_all(model=model, loss_func=loss_func, train_loader=train_loader, val_loader=val_loader, seed=seed, n=20, device=device, best_loss=best_loss)

        # 評価
        test(model=model, test_loader=test_loader, seed=seed, device=device)

        # 推論
        infer(model=model, train_dataset=train_dataset, test_dataset=test_dataset, seed=seed, device=device)


if __name__ == '__main__':
    seed_list = [25, 96, 13]
    main(seed_list=seed_list)