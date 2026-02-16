import torch
import torch.nn as nn
import numpy as np
from models.encoders import AudioEncoder
from models.encoders import TextEncoder

# 这个是使用学习的文本原型和视觉原型一起分类
class Prototype_Evolution(nn.Module):
    def __init__(self, args, text_features):
        super().__init__()

        self.args = args
        self.classes = np.arange(len(args.classnames))
        self.n_cls = len(args.classnames)
        self.text_proto = nn.Parameter(text_features, requires_grad=True)
        self.visual_proto = torch.zeros(text_features.shape[0], text_features.shape[1]).to(args.device)
        # 创建一个字典，其中每个键对应一个空的 NumPy 数组
        self.visual_features = {cls_name: np.empty((0, text_features.shape[1])) for cls_name in self.classes}



    def forward(self, audio, label, mode='train'):
        if mode=='test':
            return self.text_proto + self.visual_proto
        
        if mode=='init':    
            for ul in self.classes:
                # 选择具有相同标签的音频特征
                mask = (label == ul)
                same_label_audio = audio[mask]
                
                # 转换为 NumPy 数组
                same_label_audio_np = same_label_audio.cpu().detach().numpy()    
                self.visual_features[ul.item()] = np.vstack((self.visual_features[ul.item()], same_label_audio_np))
                
                # 更新视觉原型
                if same_label_audio_np.shape[0] > 0:
                    self.visual_proto[ul.item()] = same_label_audio.mean(dim=0)
                # import pdb; pdb.set_trace()
            
        return self.text_proto

# 获得两个模态的编码器
class CustomPENGI(nn.Module):
    def __init__(self,args,pengi):
        super().__init__()

        self.args = args
        pengi_args  = pengi.args
        self.pengi_args = pengi_args

        self.audio_encoder = AudioEncoder(
                    pengi_args.audioenc_name, pengi_args.out_emb, pengi_args.d_proj,
                    pengi_args.sampling_rate, pengi_args.window_size, pengi_args.hop_size, pengi_args.mel_bins, pengi_args.fmin, pengi_args.fmax, pengi_args.classes_num, 
                    pengi_args.specaug, pengi_args.mixup, pengi_args.use_pretrained_audioencoder, pengi_args.freeze_audio_encoder_weights,
                    pengi_args.use_precomputed_melspec, pengi_args.pretrained_audioencoder_path)

        self.text_encoder = TextEncoder(
                    pengi_args.d_proj, 
                    pengi_args.text_model, pengi_args.transformer_embed_dim,
                    pengi_args.freeze_text_encoder_weights)


        # load the weights of the pengi pre-trained audio and text encoders
        print("\n\nPALM: loading the weights of the pengi pre-trained audio and text encoders ...\n\n")
        self.audio_encoder.load_state_dict(pengi.model.audio_encoder.state_dict())
        self.text_encoder.load_state_dict(pengi.model.caption_encoder.state_dict())

        self.audio_encoder.eval()
        self.text_encoder.eval()
        self.process_text = pengi.preprocess_text
        self.device = args.device

        # 将类别标签初始化为类原型
        prompts = [f"a aduio of {class_name}" for class_name in self.args.classnames]
        tokenized_prompts = self.process_text(prompts, enc_tok=True, add_text=False)
        prompts_tokens = tokenized_prompts['input_ids']
        prompts_token_embeddings = self.text_encoder.base.embeddings.token_embedding(prompts_tokens)  # [batch_size, seq_length, embed_dim]
        prompts_attention_mask = tokenized_prompts['attention_mask']
        text = {"input_ids": prompts_tokens, "inputs_embeds": prompts_token_embeddings, "attention_mask": prompts_attention_mask}
        text_features = self.text_encoder(text) # text_features shape [n_text_prompts, 1024]
        self.text_features = text_features
        self.prompt_learner = Prototype_Evolution(args, text_features)

        


    def forward(self, audio, label, mode='test'):

        audio_features = self.audio_encoder(audio)[0] # audio_features shape [n_audio_files, 1024]
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

       # 获得原型
        text_proto = self.prompt_learner(audio_features, label, mode)
        text_proto = text_proto / text_proto.norm(dim=-1, keepdim=True)
        # visual_proto = visual_proto / visual_proto.norm(dim=-1, keepdim=True)


        logit_scale = 100.0
        logits = logit_scale * audio_features @ text_proto.t()  # logits shape [n_audio_files, n_text_prompts]
        # logits += logit_scale * audio_features @ visual_proto.t()

        return logits


    def save_features(self, audio, label):
        audio_features = self.audio_encoder(audio)[0] # audio_features shape [n_audio_files, 1024]
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        return audio_features, label, self.prompt_learner.text_proto, self.prompt_learner.visual_proto, self.args.classnames, self.text_features