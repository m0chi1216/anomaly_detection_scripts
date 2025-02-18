# 第9章 動画分類（ECO：Efficient 3DCNN）
# 9.4	Kinetics動画データセットからDataLoaderの作成

# 必要なパッケージのimport
import os
from PIL import Image
import csv
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch import nn

import torchvision

import re
import av

from torchvggish import vggish, vggish_input
from vggish import VGGish
from av.video.frame import VideoFrame
from transformers import VideoMAEImageProcessor


def make_datapath_list(root_path):
    """
    動画を画像データにしたフォルダへのファイルパスリストを作成する。
    root_path : str、データフォルダへのrootパス
    Returns：ret : video_list、動画を画像データにしたフォルダへのファイルパスリスト
    """

    # 動画を画像データにしたフォルダへのファイルパスリスト
    video_list = list()

    # root_pathにある、クラスの種類とパスを取得
    class_list = os.listdir(path=root_path)

    # 各クラスの動画ファイルを画像化したフォルダへのパスを取得
    for class_list_i in (class_list):  # クラスごとのループ

        # クラスのフォルダへのパスを取得
        class_path = os.path.join(root_path, class_list_i)

        # 各クラスのフォルダ内の画像フォルダを取得するループ
        for file_name in os.listdir(class_path):

            # ファイル名と拡張子に分割
            name, ext = os.path.splitext(file_name)

            # フォルダでないmp4ファイルは無視
            if ext == '.mp4' or name == 'audio':
                continue

            # 動画ファイルを画像に分割して保存したフォルダのパスを取得
            video_img_directory_path = os.path.join(class_path, name)

            # vieo_listに追加
            video_list.append(video_img_directory_path)

    return video_list


def make_audio_list(root_path):

    # 動画を画像データにしたフォルダへのファイルパスリスト
    audio_list = list()

    # root_pathにある、クラスの種類とパスを取得
    class_list = os.listdir(path=root_path)

    # 各クラスの動画ファイルを画像化したフォルダへのパスを取得
    for class_list_i in (class_list):  # クラスごとのループ

        # クラスのフォルダへのパスを取得
        class_path = os.path.join(root_path, class_list_i)
        class_path = os.path.join(class_path, "audio")

        # 各クラスのフォルダ内の画像フォルダを取得するループ
        for file_name in os.listdir(class_path):

            # ファイル名と拡張子に分割
            name, ext = os.path.splitext(file_name)

            if ext != '.wav':
                continue

            audio_list.append(os.path.join(class_path, file_name))

    return audio_list


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


class VideoTransform():
    """
    動画を画像にした画像ファイルの前処理クラス。学習時と推論時で異なる動作をします。
    動画を画像に分割しているため、分割された画像たちをまとめて前処理する点に注意してください。
    """

    def __init__(self, resize, crop_size, mean, std):
        self.data_transform = {
            'train': torchvision.transforms.Compose([
                # DataAugumentation()  # 今回は省略
                GroupResize(int(resize)),  # 画像をまとめてリサイズ　
                GroupCenterCrop(crop_size),  # 画像をまとめてセンタークロップ
                GroupToTensor(),  # データをPyTorchのテンソルに
                GroupImgNormalize(mean, std),  # データを標準化
                Stack()  # 複数画像をframes次元で結合させる
            ]),
            'val': torchvision.transforms.Compose([
                GroupResize(int(resize)),  # 画像をまとめてリサイズ　
                GroupCenterCrop(crop_size),  # 画像をまとめてセンタークロップ
                GroupToTensor(),  # データをPyTorchのテンソルに
                GroupImgNormalize(mean, std),  # データを標準化
                Stack()  # 複数画像をframes次元で結合させる
            ])
        }

    def __call__(self, img_group, phase):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img_group)


# 前処理で使用するクラスたちの定義


class GroupResize():
    ''' 画像をまとめてリスケールするクラス。
    画像の短い方の辺の長さがresizeに変換される。
    アスペクト比は保たれる。
    '''

    def __init__(self, resize, interpolation=Image.BILINEAR):
        '''リスケールする処理を用意'''
        self.rescaler = torchvision.transforms.Resize(resize, interpolation)

    def __call__(self, img_group):
        '''リスケールをimg_group(リスト)内の各imgに実施'''
        return [self.rescaler(img) for img in img_group]


class GroupCenterCrop():
    ''' 画像をまとめてセンタークロップするクラス。
        （crop_size, crop_size）の画像を切り出す。
    '''

    def __init__(self, crop_size):
        '''センタークロップする処理を用意'''
        self.ccrop = torchvision.transforms.CenterCrop(crop_size)

    def __call__(self, img_group):
        '''センタークロップをimg_group(リスト)内の各imgに実施'''
        return [self.ccrop(img) for img in img_group]


class GroupToTensor():
    ''' 画像をまとめてテンソル化するクラス。
    '''

    def __init__(self):
        '''テンソル化する処理を用意'''
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        '''テンソル化をimg_group(リスト)内の各imgに実施
        0から1ではなく、0から255で扱うため、255をかけ算する。
        0から255で扱うのは、学習済みデータの形式に合わせるため
        '''

        return [self.to_tensor(img)*255 for img in img_group]


class GroupImgNormalize():
    ''' 画像をまとめて標準化するクラス。
    '''

    def __init__(self, mean, std):
        '''標準化する処理を用意'''
        self.normlize = torchvision.transforms.Normalize(mean, std)

    def __call__(self, img_group):
        '''標準化をimg_group(リスト)内の各imgに実施'''
        return [self.normlize(img) for img in img_group]


class Stack():
    ''' 画像を一つのテンソルにまとめるクラス。
    '''

    def __call__(self, img_group):
        '''img_groupはtorch.Size([3, 224, 224])を要素とするリスト
        '''
        ret = torch.cat([(x.flip(dims=[0])).unsqueeze(dim=0)
                         for x in img_group], dim=0)  # frames次元で結合
        # x.flip(dims=[0])は色チャネルをRGBからBGRへと順番を変えています（元の学習データがBGRであったため）
        # unsqueeze(dim=0)はあらたにframes用の次元を作成しています

        return ret


def get_label_id_dictionary(label_dicitionary_path='./label_dict.csv'):
    label_id_dict = {}
    id_label_dict = {}

    # eoncodingはUbuntuもこれで良いのか、確認せねば
    with open(label_dicitionary_path, encoding="utf-8_sig") as f:

        # 読み込む
        reader = csv.DictReader(f, delimiter=",", quotechar='"')

        # 1行ずつ読み込み、辞書型変数に追加します
        for row in reader:
            label_id_dict.setdefault(
                row["class_label"], int(row["label_id"])-1)
            id_label_dict.setdefault(
                int(row["label_id"])-1, row["class_label"])

    return label_id_dict,  id_label_dict


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
    face_num = []
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            # frame = frame.reformat(width=frame_width, height=frame_width)
            img = frame.to_image()
            img = img.resize((int(img.width * (frame_width / img.height)), frame_width))
            img = crop_center(img, frame_width, frame_width)
            frame = VideoFrame.from_image(img)
            # print(f"frame : {frame}", flush=True)
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


# class VideoDataset(torch.utils.data.Dataset):
#     """
#     動画のDataset
#     """

#     def __init__(self, video_list, audio_list):
#         self.video_list = video_list  # 動画画像のフォルダへのパスリスト
#         self.audio_list = audio_list

#     def __len__(self):
#         '''動画の数を返す'''
#         return len(self.video_list)

#     def __getitem__(self, index):
#         video_dir_path = self.video_list[index]  # 画像が格納されたフォルダ
#         if 'abnormal' in video_dir_path:
#             label_id = 1
#         elif 'normal' in video_dir_path:
#             label_id = 0

#         # 3. 前処理を実施
#         # imgs_transformed = self.transform(img_group, phase=self.phase)
#         container = av.open(video_dir_path)
#         clip_len = 16
#         indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=int(container.streams.video[0].frames / clip_len), seg_len=container.streams.video[0].frames)
#         video = read_video_pyav(container, indices, frame_width=224)

#         # print(f"vide oshape: {video.shape}", flush=True)

#         image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
#         video_embeddings = image_processor(list(video), return_tensors="pt")
#         video_embeddings = list(video_embeddings.data.values())[0]

#         audio_dir_path = self.audio_list[index]

#         audio_embeddings = vggish_input.wavfile_to_examples(audio_dir_path)
#         # print(f"audio shape: {audio_embeddings.shape}", flush=True)
#         if (audio_embeddings.shape[0] > 10):
#             audio_embeddings = audio_embeddings[-10:]
#         elif (audio_embeddings.shape[0] < 10):
#             while audio_embeddings.shape[0] != 10:
#                 audio_embeddings = torch.cat([audio_embeddings[:1], audio_embeddings], dim=0)

#         audio_embeddings = audio_embeddings.detach()

#         # print(f"audio shape: {audio_embeddings.shape}", flush=True)

#         # with open('./log/all_video.log', 'a', encoding='utf-8') as f:
#         #     f.write(f'{video_dir_path}, {audio_dir_path}\n')

#         return video_embeddings, audio_embeddings, label_id


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_list, audio_list):
        self.label_df = pd.read_csv('./data/annotation_result3.csv')
        valid_video_list = []
        valid_audio_list = []
        label_list = []
        reason_list = []
        file_list = []
        for i in range(len(video_list)):
            if self.label_df.iloc[i, 8]:
                valid_video_list.append(video_list[i])
                valid_audio_list.append(audio_list[i])
                label = int(self.label_df.iloc[i, 9])
                label_list.append(label)
                reason = str(self.label_df.iloc[i, 10])
                reason_list.append(reason)
                file_list.append(self.label_df.iloc[i, 0])

        self.video_list = valid_video_list
        self.audio_list = valid_audio_list
        self.label_list = label_list
        self.reason_list = reason_list
        self.file_list = file_list

    def __len__(self):
        '''動画の数を返す'''
        return len(self.video_list)

    def __getitem__(self, index):
        video_dir_path = self.video_list[index]
        label_id = self.label_list[index]
        reason = self.reason_list[index]
        file_name = self.file_list[index]

        # 3. 前処理を実施
        # imgs_transformed = self.transform(img_group, phase=self.phase)
        container = av.open(video_dir_path)
        clip_len = 16
        indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=int(container.streams.video[0].frames / clip_len), seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container, indices, frame_width=224)

        # print(f"vide oshape: {video.shape}", flush=True)

        image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        video_embeddings = image_processor(list(video), return_tensors="pt")
        video_embeddings = list(video_embeddings.data.values())[0]

        audio_dir_path = self.audio_list[index]

        audio_embeddings = vggish_input.wavfile_to_examples(audio_dir_path)
        # print(f"audio shape: {audio_embeddings.shape}", flush=True)
        if (audio_embeddings.shape[0] > 10):
            audio_embeddings = audio_embeddings[-10:]
        elif (audio_embeddings.shape[0] < 10):
            while audio_embeddings.shape[0] != 10:
                audio_embeddings = torch.cat([audio_embeddings[:1], audio_embeddings], dim=0)

        audio_embeddings = audio_embeddings.detach()

        # print(f"audio shape: {audio_embeddings.shape}", flush=True)

        # with open('./log/all_video.log', 'a', encoding='utf-8') as f:
        #     f.write(f'{video_dir_path}, {audio_dir_path}\n')

        return video_embeddings, audio_embeddings, label_id