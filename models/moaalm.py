import math
import numpy as np
import torch
import torch.nn as nn


from .encoders import AudioEncoder
from .encoders import TextEncoder
import torch.nn.functional as F






class Adapter(nn.Module):
    def __init__(self, input_dim, reduction_factor=4):
        super(Adapter, self).__init__()
        self.downsample = nn.Linear(input_dim, input_dim // reduction_factor)
        self.activation = nn.Tanh()
        self.upsample = nn.Linear(input_dim // reduction_factor, input_dim)

    def forward(self, x):
        residual = x
        x = self.downsample(x)
        x = self.activation(x)
        x = self.upsample(x)
        return x + residual  # 使用残差连接
        



class WeightMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))  # 输出单个权重值
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)
        
    
    def forward(self, x):
        # 输入x形状: (batch_size, 1024)
        return self.mlp(x)


class MixtureOfAdapters(nn.Module):
    def __init__(self, args):
        super(MixtureOfAdapters, self).__init__()
        self.args = args
        self.input_dim = args.ctx_dim
        self.text = Adapter(self.input_dim)
        self.audio = Adapter(self.input_dim)
        self.weight = WeightMLP(self.input_dim)

        
    def forward(self,  audios, texts):
        audio_features = self.audio(audios)
        text_features = self.text(texts)
        # 结合所有输出
        # output = self.out(torch.cat([audio_features, text_features], dim=-1))
        a = self.weight(audio_features)
        b = self.weight(text_features)
        output = audio_features * a + text_features *b

        return output  

class PromptLearner(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        classnames = args.classnames
        n_cls = len(classnames)

        ctx_dim = args.ctx_dim 

        print("Initializing a generic context")
        ctx = torch.empty(n_cls, ctx_dim)
        torch.nn.init.normal_(ctx, std=0.02)
        self.ctx = torch.nn.Parameter(ctx)


        self.n_cls = n_cls
        self.lambdas = nn.Parameter(torch.rand(n_cls))
        
        self.moa = MixtureOfAdapters(args).to(args.device)
        self.reference = nn.Linear(ctx_dim, n_cls, bias=True).to(args.device)
        nn.init.orthogonal_(self.reference.weight)
        nn.init.constant_(self.reference.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        
    def Transformation_Matrix(self, prototype):
        C = prototype # (N, emd_dim)
        eps = 1e-6 #避免除以0的情况
        R = self.reference.weight # (emd_dim, N)

        # 标准化
        power_R = ((R * R).sum(dim=1, keepdim=True)).sqrt()
        R = R / (power_R + eps)

        # 标准化
        power_C = ((C * C).sum(dim=1, keepdim=True)).sqrt()
        C = C / (power_C + eps)
        P = torch.matmul(torch.pinverse(C), R)
        P = P.permute(1, 0)
        return P


    def forward(self, audio_features, text_features, prototypes):

        lambdas = torch.sigmoid(self.lambdas).reshape(-1,1)  # [n_cls, 1]
        
        updated_text_features = (1-lambdas)*text_features + (lambdas*self.ctx)     # [n_text_prompts, 1024]
        updated_text_features = updated_text_features / updated_text_features.norm(dim=-1, keepdim=True)
        
        
        text_features = self.moa(prototypes, updated_text_features)
        
        # audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # P = self.Transformation_Matrix(text_features)
        # weight = P.view(P.size(0), P.size(1), 1)
        # text_features = F.conv1d(text_features.squeeze(0).unsqueeze(2), weight).squeeze(2)
        # audio_features = F.conv1d(audio_features.squeeze(0).unsqueeze(2), weight).squeeze(2)

        return text_features, audio_features

class CustomPENGI(nn.Module):
    def __init__(self, args, pengi):
        super().__init__()

        self.args = args
        pengi_args  = pengi.args
        self.pengi_args = pengi_args
        
        pengi_args.specaug = args.spec_aug

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
        print("\n\nCOOP: loading the weights of the pengi pre-trained audio and text encoders ...\n\n")
        self.audio_encoder.load_state_dict(pengi.model.audio_encoder.state_dict())
        self.text_encoder.load_state_dict(pengi.model.caption_encoder.state_dict())

        self.audio_encoder.eval()
        self.text_encoder.eval()


        self.prompt_learner = PromptLearner(args)
        self.process_text = pengi.preprocess_text

        self.device = args.device
        
        classnames = args.classnames
        n_cls = len(classnames)
        self.classes = range(n_cls)
        ctx_dim = args.ctx_dim # 512
        self.memory = {cls_name: np.empty((0, ctx_dim)) for cls_name in self.classes}
        self.old_prototypes = np.zeros((n_cls, args.ctx_dim))
        ctx_vectors = torch.empty(n_cls, ctx_dim)
        torch.nn.init.normal_(ctx_vectors, std=0.02)                
        self.learned_prototypes  = torch.nn.Parameter(ctx_vectors).to(args.device)
        self.sampler = Sampler(args)
    
    
    
    
    
    
    def add_memory(self, label, features):    
        features = self.audio_encoder(features)[0]
        for ul in self.classes:
            # 选择具有相同标签的音频特征
            mask = (label == ul)
            same_label_audio = features[mask]
            # 转换为 NumPy 数组
            same_label_audio_np = same_label_audio.cpu().detach().numpy()    
            self.memory[ul] = np.vstack((self.memory[ul], same_label_audio_np))
            # 更新视觉原型
            if self.memory[ul].shape[0] > 0:
                self.old_prototypes[ul] = self.memory[ul].mean(axis=0)
    
    
    
    # 先做一个原型的初始化
    def forward(self, audio):

        audio_features = self.audio_encoder(audio)[0] # audio_features shape [n_audio_files, 1024]        
        
        prototypes = torch.tensor(self.old_prototypes, dtype=torch.float32).to(self.args.device) + self.learned_prototypes
        
        prompts = [f"{class_name}" for class_name in self.args.classnames]
        tokenized_prompts = self.process_text(prompts, enc_tok=True, add_text=False)
        prompts_tokens = tokenized_prompts['input_ids'].to(self.device) 
        prompts_attention_mask = tokenized_prompts['attention_mask'].to(self.device) 
        
        with torch.no_grad():
            prompts_token_embeddings = self.text_encoder.base.embeddings.token_embedding(prompts_tokens)   # [batch_size, seq_length, embed_dim]
        
        text = {"input_ids": prompts_tokens, "inputs_embeds": prompts_token_embeddings, "attention_mask": prompts_attention_mask}
        text_features = self.text_encoder(text) # text_features shape [n_text_prompts, 1024]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        text_features, audio_features = self.prompt_learner(audio_features, text_features, prototypes) # text_features shape [n_text_prompts, 1024]
        
        
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = 100.0
        logits = logit_scale * audio_features @ text_features.t()  # logits shape [n_audio_files, n_text_prompts]
        
        
        # breakpoint()

        return logits
    


class Sampler(nn.Module):
    def __init__(self, args):
        super(Sampler, self).__init__()
        self.args = args
        self.dim = args.ctx_dim
        # TOP R
        self.k = 5
        # the number of samples per shot
        self.num_sampled = 20
        self.threshold = 0.8

    def calculate_var(self, features):
        v_mean = features.mean(dim=1) 
        v_cov = []
        for i in range(features.shape[0]):
            diag = torch.var(features[i], dim=0)
            v_cov.append(diag)
        v_cov = torch.stack(v_cov)

        return v_mean, v_cov

    def forward(self, prototypes, queries):
        self.nway = prototypes.shape[0]
        self.kshot = 5
        similarity = prototypes / prototypes.norm(dim=-1, keepdim=True) @ (queries / queries.norm(dim=-1, keepdim=True)).t()
        # (N, K, NQ)
        similarity = -similarity.view(prototypes.shape[0], prototypes.shape[1], -1)

        values, indices = similarity.topk(self.k, dim=2, largest=False, sorted=True)     
        nindices = indices.view(-1, self.k)
       
        convex_feat = []
        for i in range(nindices.shape[0]):
            convex_feat.append(queries.index_select(0, nindices[i]))
        convex_feat = torch.stack(convex_feat) # NK, k, 768

        sampled_data = convex_feat.view(prototypes.shape[0], self.kshot * self.k, self.dim)
       
        return sampled_data

