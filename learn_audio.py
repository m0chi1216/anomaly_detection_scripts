import av
import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
import sys
import glob
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from av.video.frame import VideoFrame
from transformers import AutoImageProcessor, TimesformerForVideoClassification, VideoMAEImageProcessor, VideoMAEForVideoClassification, VivitImageProcessor, VivitForVideoClassification

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve

from dataloader_multimodal import natural_keys

from torchvggish import vggish, vggish_input
from vggish import VGGish

from torchmetrics.classification import PrecisionRecallCurve

args = sys.argv
CKPT_NAME = args[1]

warnings.filterwarnings("ignore")

np.random.seed(0)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, audio_list):
        self.label_df = pd.read_csv('./data/annotation_result2.csv')
        valid_list = []
        label_list = []
        for i in range(len(audio_list)):
            if self.label_df.iloc[i, 8]:
                valid_list.append(audio_list[i])
                label = int(self.label_df.iloc[i, 9])
                label_list.append(label)
        
        self.audio_list = valid_list
        self.label_list = label_list

    def __len__(self):
        '''動画の数を返す'''
        return len(self.audio_list)

    def __getitem__(self, index):
        dir_path = self.audio_list[index]
        label_id = self.label_list[index]

        embeddings = vggish_input.wavfile_to_examples(dir_path)
        if (embeddings.shape[0] > 10):
            embeddings = embeddings[-10:]
        elif (embeddings.shape[0] < 10):
            while embeddings.shape[0] != 10:
                embeddings = torch.cat([embeddings[:1], embeddings], dim=0)

        embeddings = embeddings.detach()
        # print(f"embedding size: {embeddings.shape}", flush=True)
        return embeddings, label_id


def load_pretrained_weights(model_dict, pretrained_model_dict):

    # 現在のネットワークモデルのパラメータ名
    param_names = []  # パラメータの名前を格納していく
    for name, param in model_dict.items():
        param_names.append(name)

    # 現在のネットワークの情報をコピーして新たなstate_dictを作成
    new_state_dict = model_dict.copy()

    # 新たなstate_dictに学習済みの値を代入
    print("学習済みのパラメータをロードします")
    for index, (key_name, value) in enumerate(pretrained_model_dict.items()):
        name = param_names[index]  # 現在のネットワークでのパラメータ名を取得
        new_state_dict[name] = value  # 値を入れる

        # 何から何にロードされたのかを表示
        # print(str(key_name)+"→"+str(name), flush=True)

    return new_state_dict


class VggishModel(nn.Module):
    def __init__(self):
        super(VggishModel, self).__init__()

        # Vggish
        self.vggish = VGGish()

        # self.conv1d = nn.Conv1d(10, 1, 1)
        self.fc_middle = nn.Linear(in_features=1280, out_features=512, bias=True)

        # クラス分類の全結合層
        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x):

        bs = x.shape[0]
        out = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        out = self.vggish(out)

        # バッチに戻す
        out = out.view(bs, x.shape[1], 128)

        # out = self.conv1d(out)

        # [batch_size, 1280]に変換
        out = torch.flatten(out, 1)

        out = self.fc_middle(out)

        out = self.fc_final(out)

        return out


def train(model, train_loader, val_loader, seed, n=10, device="cuda:0"):
    class_weight = [1.0, 2.86]
    weights = torch.tensor(class_weight).to(device)
    print(f'set weight loss: {class_weight}', flush=True)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    # criterion = nn.CrossEntropyLoss().to(device)

    # optimizer = optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.9)
    optimizer = optim.SGD(params=model.parameters(), lr=0.0002, momentum=0.9)

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

        if val_loss_value < best_loss:
            stop_count = 0
            # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
            torch.save(model.state_dict(), f'./checkpoints/vggish/{CKPT_NAME}-{seed}-best.pth')
            best_loss = val_loss_value
            print('save best model', flush=True)
        else:
            stop_count += 1
            if stop_count >= early_stop:
                torch.save(model.state_dict(), f'./checkpoints/vggish/{CKPT_NAME}-{seed}-last.pth')
                print('early stopping', flush=True)
                return
        if epoch == n-1:
            torch.save(model.state_dict(), f'./checkpoints/vggish/{CKPT_NAME}-{seed}-last.pth')
            


def test(model, test_loader, seed, device="cuda:0"):
    # file_name = re.findall(r'(.+).py', os.path.basename(__file__))[0]
    model.load_state_dict(torch.load( f'./checkpoints/vggish/{CKPT_NAME}-{seed}-best.pth'))

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
    # np.save(f'./analysis/vggish/p-{seed}.npy', precision)
    # np.save(f'./analysis/vggish/r-{seed}.npy', recall)
    # np.save(f'./analysis/vggish/th-{seed}.npy', thresholds)

    # np.save(f'./analysis/vggish/preds-{seed}.npy', np.array(preds_all))


def main(seed_list=[42]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audio_list = glob.glob('./data/audio/*.wav')
    audio_list = sorted(audio_list, key=natural_keys)

    for seed in seed_list:
        print(f'\n\n--- SEED={seed} ---', flush=True)

        all_dataset = MyDataset(audio_list)

        all_len = len(all_dataset)
        train_len = int(0.7 * all_len)
        val_len = int((all_len-train_len) / 2)
        test_len = all_len - train_len - val_len

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset=all_dataset, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))


        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

        print(f'all: {len(all_dataset)}, train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}', flush=True)

        model = VggishModel()

        pretrained_model = vggish()
        pretrained_model_dict = pretrained_model.state_dict()

        # 現在のモデルの変数名などを取得
        model_dict = model.state_dict()

        # 学習済みモデルのstate_dictを取得
        new_state_dict = load_pretrained_weights(model_dict, pretrained_model_dict)

        # 学習済みモデルのパラメータを代入
        model.load_state_dict(new_state_dict)

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