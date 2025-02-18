import torch
import torch.nn as nn

from vggish import VGGish
from torchvggish import vggish

from av.video.frame import VideoFrame
from transformers import  VideoMAEForVideoClassification, VideoMAEForPreTraining, AutoFeatureExtractor, ASTForAudioClassification


def load_pretrained_weights(model_dict, pretrained_model_dict, model_name):

    # 現在のネットワークモデルのパラメータ名
    param_names = []  # パラメータの名前を格納していく
    for name, param in model_dict.items():
        param_names.append(name)

    # 現在のネットワークの情報をコピーして新たなstate_dictを作成
    new_state_dict = model_dict.copy()

    # 新たなstate_dictに学習済みの値を代入
    print(f"学習済みのパラメータをロードします model: {model_name}", flush=True)
    for index, (key_name, value) in enumerate(pretrained_model_dict.items()):
        if model_name == 'eco':
            if 'fc' in str(key_name):
                continue
        elif model_name == 'vggish':
            if 'fc' in str(key_name):
                continue
        elif model_name == 'mae':
            if 'fc' in str(key_name):
                continue

        for i in range(len(param_names)): 
            if key_name == param_names[i]:
                name = param_names[i]  # 現在のネットワークでのパラメータ名を取得
                new_state_dict[name] = value  # 値を入れる
                # 何から何にロードされたのかを表示
                # print(str(key_name)+"→"+str(name), flush=True)
                break

    return new_state_dict


class DecisionModel(nn.Module):
    def __init__(self):
        super(DecisionModel, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.mae_pretrain = VideoMAEModel()
        self.mae_pretrain.load_state_dict(torch.load('./checkpoints/mae/learn_videomae-42-best.pth'))
        for param in self.mae_pretrain.parameters():
            param.requires_grad = False
        
        self.vggish_pretrain = ACModel()
        self.vggish_pretrain.load_state_dict(torch.load('./checkpoints/vggish/learn_audio-42-best.pth'))
        for param in self.vggish_pretrain.parameters():
            param.requires_grad = False

        # mae_model = VideoMAEFeatureModel()
        # self.mae_model = mae_model


        # vggish_model = AudioFeatureModel()
        # self.vggish_model = vggish_model

        # self.mha1 = nn.MultiheadAttention(512, 2)

        # self.mha2 = nn.MultiheadAttention(512, 2)

        # self.fusion_layer = nn.Linear(in_features=1024, out_features=512, bias=True)

        # self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

        self.cma_model = CMAModel()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):

        # out_video = self.mae_model(x1)
        # out_audio = self.vggish_model(x2)

        # attn_output_v, attn_weights_v = self.mha1(out_video, out_audio, out_audio)
        # attn_output_a, attn_weights_a = self.mha2(out_audio, out_video, out_video)

        # out_video = attn_output_v + out_video
        # out_audio = attn_output_a + out_audio

        # out = torch.cat([out_video, out_audio], dim=1)

        # out = self.fusion_layer(out)

        # out = self.fc_final(out)
        out = self.cma_model(x1, x2)

        pred_weight = self.softmax(out)

        video_pretrain_out = self.mae_pretrain(x1)
        video_pretrain_out = self.softmax(video_pretrain_out)
        out_v = video_pretrain_out * pred_weight[:,0].unsqueeze(1)

        audio_pretrain_out = self.vggish_pretrain(x2)
        audio_pretrain_out = self.softmax(audio_pretrain_out)
        out_a = audio_pretrain_out * pred_weight[:,1].unsqueeze(1)

        out = out_v + out_a

        return out


class ACModel(nn.Module):
    def __init__(self):
        super(ACModel, self).__init__()

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

        out = torch.flatten(out, 1)

        out = self.fc_middle(out)

        out = self.fc_final(out)

        return out
    

class AudioFeatureModel(nn.Module):
    def __init__(self):
        super(AudioFeatureModel, self).__init__()

        # Vggish
        self.vggish = VGGish()

        pretrained_model = vggish()
        pretrained_model_dict = pretrained_model.state_dict()

        # 現在のモデルの変数名などを取得
        model_dict = self.vggish.state_dict()

        # 学習済みモデルのstate_dictを取得
        new_state_dict = load_pretrained_weights(model_dict, pretrained_model_dict, "vggish")

        # 学習済みモデルのパラメータを代入
        self.vggish.load_state_dict(new_state_dict)

        # self.conv1d = nn.Conv1d(10, 1, 1)
        self.fc_middle = nn.Linear(in_features=1280, out_features=512, bias=True)

        self.fc_final = nn.Linear(in_features=512, out_features=512, bias=True)

    def forward(self, x):

        bs = x.shape[0]
        out = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        out = self.vggish(out)

        # バッチに戻す
        out = out.view(bs, x.shape[1], 128)

        out = torch.flatten(out, 1)

        out = self.fc_middle(out)

        out = self.fc_final(out)

        return out


class VideoMAEModel(nn.Module):
    def __init__(self):
        super(VideoMAEModel, self).__init__()

        # print(f'model: {model_name}', flush=True)
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        # model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
        # model.load_state_dict(torch.load( f'./checkpoints/mae/{CKPT_NAME}-{seed}-best.pth'))

        model.classifier = torch.nn.Linear(768, 768)
        self.fc_middle = nn.Linear(in_features=768, out_features=512, bias=True)

        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

        self.main_model = model

    def forward(self, x):

        bs = x.shape[0]
        out = x.view(bs, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
        out = self.main_model(out)

        out = out.logits
        out = self.fc_middle(out)
        out = self.fc_final(out)

        return out
    

class VideoMAEFeatureModel(nn.Module):
    def __init__(self):
        super(VideoMAEFeatureModel, self).__init__()

        # print(f'model: {model_name}', flush=True)
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

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


class VideoMAEPretrainModel(nn.Module):
    def __init__(self):
        super(VideoMAEPretrainModel, self).__init__()

        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

        model_state = model.state_dict()
        pretrain_state = torch.load( f'./checkpoints/mae-pretrain/mae-pretrain-6-e29.pth')
        state = {k:v for k, v in pretrain_state.items() if k in model_state and v.size() == model_state[k].size()}

        load_state = {}
        for k, v in model_state.items():
            if k in pretrain_state and v.size() == pretrain_state[k].size():
                load_state[k] = pretrain_state[k]
            else:
                load_state[k] = v

        model = model.load_state_dict(load_state)

        model.classifier = torch.nn.Linear(768, 768)
        self.fc_middle = nn.Linear(in_features=768, out_features=512, bias=True)

        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

        self.main_model = model

    def forward(self, x):

        bs = x.shape[0]
        out = x.view(bs, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
        out = self.main_model(out)

        out = out.logits
        out = self.fc_middle(out)
        out = self.fc_final(out)

        return out


class CMAModel(nn.Module):
    def __init__(self, seed=25, is_freeze=True):
        super(CMAModel, self).__init__()

        print('-- load CMAModel --', flush=True)
        mae_model = VideoMAEFeatureModel()

        self.mae_model = mae_model

        # self.mae_pretrain = VideoMAEModel()
        # self.mae_pretrain.load_state_dict(torch.load('./checkpoints/mae/learn_videomae-25-best.pth'))
        mae_pretrain = torch.load(f'./checkpoints/videomae/softmax-{seed}-best.pth')

        new_state_dict = load_pretrained_weights(self.mae_model.state_dict(), mae_pretrain, "mae")
        self.mae_model.load_state_dict(new_state_dict)

        
        vggish_model = AudioFeatureModel()

        self.vggish_model = vggish_model

        vggish_pretrain = torch.load(f'./checkpoints/vggish/softmax-{seed}-best.pth')

        new_state_dict = load_pretrained_weights(self.vggish_model.state_dict(), vggish_pretrain, "vggish")
        self.vggish_model.load_state_dict(new_state_dict)

        if is_freeze:
            print('-- freeze encoder parameter --', flush=True)
            for param in self.mae_model.parameters():
                param.requires_grad = False
            for param in self.vggish_model.parameters():
                param.requires_grad = False

        self.mha1 = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        self.mha2 = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        self.fusion_layer = nn.Linear(in_features=1024, out_features=512, bias=True)

        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x1, x2):

        out_video = self.mae_model(x1)
        out_audio = self.vggish_model(x2)

        # print(f'video shape: {out_video.shape}', flush=True)
        # print(f'audio shape: {out_audio.shape}', flush=True)

        attn_output_v, attn_weights_v = self.mha1(out_video, out_audio, out_audio)
        attn_output_a, attn_weights_a = self.mha2(out_audio, out_video, out_video)

        out_video = attn_output_v + out_video
        out_audio = attn_output_a + out_audio

        out = torch.cat([out_video, out_audio], dim=1)

        out = self.fusion_layer(out)

        out = self.fc_final(out)

        return out
    

class CMAFeatureModel(nn.Module):
    def __init__(self, seed=25, is_freeze=True):
        super(CMAFeatureModel, self).__init__()

        mae_model = VideoMAEFeatureModel()
        self.mae_model = mae_model

        mae_pretrain = torch.load(f'./checkpoints/mae-metric/rank-{seed}-best.pth')

        new_state_dict = load_pretrained_weights(self.mae_model.state_dict(), mae_pretrain, "feature")
        self.mae_model.load_state_dict(new_state_dict)

        vggish_model = AudioFeatureModel()
        self.vggish_model = vggish_model

        vggish_pretrain = torch.load(f'./checkpoints/vggish-metric/rank-{seed}-best.pth')

        new_state_dict = load_pretrained_weights(self.vggish_model.state_dict(), vggish_pretrain, "feature")
        self.vggish_model.load_state_dict(new_state_dict)

        if is_freeze:
            print('-- freeze encoder parameter --', flush=True)
            for param in self.mae_model.parameters():
                param.requires_grad = False
            for param in self.vggish_model.parameters():
                param.requires_grad = False

        self.mha1 = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        self.mha2 = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        self.fusion_layer = nn.Linear(in_features=1024, out_features=512, bias=True)

        self.fc_final = nn.Linear(in_features=512, out_features=512, bias=True)

    def forward(self, x1, x2):

        out_video = self.mae_model(x1)
        out_audio = self.vggish_model(x2)

        attn_output_v, attn_weights_v = self.mha1(out_video, out_audio, out_audio)
        attn_output_a, attn_weights_a = self.mha2(out_audio, out_video, out_video)

        out_video = attn_output_v + out_video
        out_audio = attn_output_a + out_audio

        out = torch.cat([out_video, out_audio], dim=1)

        out = self.fusion_layer(out)

        out = self.fc_final(out)

        return out


class ConcatModel(nn.Module):
    def __init__(self, seed=25, is_freeze=True):
        super(ConcatModel, self).__init__()

        print('-- load ConcatModel --', flush=True)
        mae_model = VideoMAEFeatureModel()

        self.mae_model = mae_model

        # self.mae_pretrain = VideoMAEModel()
        # self.mae_pretrain.load_state_dict(torch.load('./checkpoints/mae/learn_videomae-25-best.pth'))
        mae_pretrain = torch.load(f'./checkpoints/mae/learn_videomae-{seed}-best.pth')

        new_state_dict = load_pretrained_weights(self.mae_model.state_dict(), mae_pretrain, "mae")
        self.mae_model.load_state_dict(new_state_dict)

        vggish_model = AudioFeatureModel()

        self.vggish_model = vggish_model

        vggish_pretrain = torch.load(f'./checkpoints/vggish/learn_audio-{seed}-best.pth')

        new_state_dict = load_pretrained_weights(self.vggish_model.state_dict(), vggish_pretrain, "vggish")
        self.vggish_model.load_state_dict(new_state_dict)

        if is_freeze:
            print('-- freeze encoder parameter --', flush=True)
            for param in self.mae_model.parameters():
                param.requires_grad = False
            for param in self.vggish_model.parameters():
                param.requires_grad = False

        self.fusion_layer = nn.Linear(in_features=1024, out_features=512, bias=True)

        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x1, x2):

        out_video = self.mae_model(x1)
        out_audio = self.vggish_model(x2)

        out = torch.cat([out_video, out_audio], dim=1)

        out = self.fusion_layer(out)

        out = self.fc_final(out)

        return out
    

class ASTModel(nn.Module):
    def __init__(self):
        super(ASTModel, self).__init__()

        # print(f'model: {model_name}', flush=True)
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        # model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
        # model.load_state_dict(torch.load( f'./checkpoints/mae/{CKPT_NAME}-{seed}-best.pth'))

        model.classifier = torch.nn.Linear(768, 512)
        self.fc_middle = nn.Linear(in_features=512, out_features=512, bias=True)

        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

        self.main_model = model

    def forward(self, x):

        bs = x.shape[0]
        out = x.view(bs, x.shape[-2], x.shape[-1])
        out = self.main_model(out)

        out = out.logits
        out = self.fc_middle(out)
        out = self.fc_final(out)

        return out