import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from .encoders import AudioEncoder
from .encoders import TextEncoder


# ===================== 1. 外部频域特征提取模块（仅提取，不增强） =====================
class ExternalMelExtractor(nn.Module):
    """外部提取梅尔谱频域特征（仅做基础提取，无增强，交给PromptLearner处理）"""
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=50,
            f_max=8000
        )
        self.n_mels = n_mels

    def forward(self, audio_waveform):
        # 提取梅尔谱并池化为向量 [B, n_mels]
        mel = self.mel_spectrogram(audio_waveform)
        mel = torch.mean(mel, dim=-1)  # [B, n_mels]
        mel = mel / (mel.norm(dim=-1, keepdim=True) + 1e-8)
        return mel


# ===================== 2. 重构PromptLearner（融入频域提示，核心改动） =====================
class PromptLearner(nn.Module):
    def __init__(self, args, n_freq_bins=128):
        super().__init__()
        self.args = args
        classnames = args.classnames
        n_cls = len(classnames)
        ctx_dim = args.ctx_dim  # 1024维（文本/音频特征维度）

        # ===== 原有可学习上下文提示（语义提示） =====
        print("Initializing a generic context")
        self.ctx_semantic = nn.Parameter(torch.empty(n_cls, ctx_dim))  # 语义提示向量
        torch.nn.init.normal_(self.ctx_semantic, std=0.02)

        # ===== 新增：频域提示编码层（将频域特征转为提示向量） =====
        self.freq_prompt_encoder = nn.Sequential(
            nn.Linear(n_freq_bins, ctx_dim),  # 频域特征→提示维度
            nn.LayerNorm(ctx_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ctx_dim, ctx_dim)       # 适配语义提示维度
        )
        # 频域提示权重（自适应学习频域提示的重要性）
        self.alpha_freq = nn.Parameter(torch.rand(n_cls, 1))  # [n_cls, 1]

        # ===== 原有融合系数 =====
        self.lambdas = nn.Parameter(torch.rand(n_cls))
        self.n_cls = n_cls
        self.ctx_dim = ctx_dim

    def forward(self, audio_features, text_features, freq_features):
        """
        输入：
        - audio_features: [B, ctx_dim] 原有音频特征（无修改）
        - text_features: [n_cls, ctx_dim] 文本特征
        - freq_features: [B, n_freq_bins] 外部提取的频域特征（梅尔谱）
        输出：
        - updated_text_features: [n_cls, ctx_dim] 融合频域提示的文本特征
        """
        # Step 1: 编码频域特征为频域提示向量 [B, ctx_dim]
        freq_prompt = self.freq_prompt_encoder(freq_features)  # 频域→提示维度
        freq_prompt = freq_prompt / (freq_prompt.norm(dim=-1, keepdim=True) + 1e-8)

        # Step 2: 聚合批次级频域提示→类别级（适配文本特征维度 [n_cls, ctx_dim]）
        # 方式：按类别对音频-频域提示加权平均（假设audio_features第一维是类别对齐的）
        # 简化版：取批次均值，适配小样本场景
        freq_prompt_cls = torch.mean(freq_prompt, dim=0, keepdim=True).repeat(self.n_cls, 1)  # [n_cls, ctx_dim]
        
        # # Step 3: 融合语义提示 + 频域提示（双重提示）
        # alpha = torch.sigmoid(self.alpha_freq)  # 频域提示权重 [n_cls, 1]
        # ctx_fusion = (1 - alpha) * self.ctx_semantic + alpha * freq_prompt_cls  # 双重提示融合
        # ctx_fusion = ctx_fusion / (ctx_fusion.norm(dim=-1, keepdim=True) + 1e-8)

        # Step 4: 原有文本特征+融合提示更新（核心逻辑不变，仅替换为双重提示）
        lambdas = torch.sigmoid(self.lambdas).reshape(-1, 1)  # [n_cls, 1]
        updated_text_features = (1 - lambdas) * text_features + lambdas * freq_prompt_cls
        updated_text_features = updated_text_features / (updated_text_features.norm(dim=-1, keepdim=True) + 1e-8)

        return updated_text_features


# ===================== 3. CustomPENGI（仅传频域特征给PromptLearner，无独立增强模块） =====================
class CustomPENGI(nn.Module):
    def __init__(self, args, pengi):
        super().__init__()
        self.args = args
        pengi_args = pengi.args
        self.pengi_args = pengi_args

        # ===== 原有代码完全保留，无修改 =====
        self.audio_encoder = AudioEncoder(
            pengi_args.audioenc_name, pengi_args.out_emb, pengi_args.d_proj,
            pengi_args.sampling_rate, pengi_args.window_size, pengi_args.hop_size, pengi_args.mel_bins, pengi_args.fmin, pengi_args.fmax, pengi_args.classes_num,
            pengi_args.specaug, pengi_args.mixup, pengi_args.use_pretrained_audioencoder, pengi_args.freeze_audio_encoder_weights,
            pengi_args.use_precomputed_melspec, pengi_args.pretrained_audioencoder_path)

        self.text_encoder = TextEncoder(
            pengi_args.d_proj,
            pengi_args.text_model, pengi_args.transformer_embed_dim,
            pengi_args.freeze_text_encoder_weights)

        print("\n\nPALM: loading the weights of the pengi pre-trained audio and text encoders ...\n\n")
        self.audio_encoder.load_state_dict(pengi.model.audio_encoder.state_dict())
        self.text_encoder.load_state_dict(pengi.model.caption_encoder.state_dict())

        self.audio_encoder.eval()
        self.text_encoder.eval()

        # ===== 初始化PromptLearner（传入频域维度） =====
        self.prompt_learner = PromptLearner(args, n_freq_bins=pengi_args.mel_bins)

        self.process_text = pengi.preprocess_text
        self.device = args.device

        # ===== 仅保留外部梅尔谱提取器（无增强模块） =====
        self.mel_extractor = ExternalMelExtractor(
            sample_rate=pengi_args.sampling_rate,
            n_fft=pengi_args.window_size,
            hop_length=pengi_args.hop_size,
            n_mels=pengi_args.mel_bins
        ).to(args.device)

    def forward(self, audio):
        # ===== 1. 原有音频特征提取（完全不变） =====
        audio_features = self.audio_encoder(audio)[0]  # [B, 1024]
        audio_features = audio_features / (audio_features.norm(dim=-1, keepdim=True) + 1e-8)

        # ===== 2. 提取频域特征（仅提取，无增强） =====
        freq_features = self.mel_extractor(audio)  # [B, n_mels]

        # ===== 3. 原有文本特征提取（完全不变） =====
        prompts = [f"{class_name}" for class_name in self.args.classnames]
        tokenized_prompts = self.process_text(prompts, enc_tok=True, add_text=False)
        prompts_tokens = tokenized_prompts['input_ids'].to(self.device)
        prompts_attention_mask = tokenized_prompts['attention_mask'].to(self.device)

        with torch.no_grad():
            prompts_token_embeddings = self.text_encoder.base.embeddings.token_embedding(prompts_tokens)

        text = {"input_ids": prompts_tokens, "inputs_embeds": prompts_token_embeddings, "attention_mask": prompts_attention_mask}
        text_features = self.text_encoder(text)  # [n_cls, 1024]
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)

        # ===== 4. Prompt学习（传入频域特征，以提示形式融合） =====
        text_features = self.prompt_learner(audio_features, text_features, freq_features)

        # ===== 5. 原有Logits计算（完全不变） =====
        logit_scale = 100.0
        logits = logit_scale * audio_features @ text_features.t()  # [B, n_cls]

        return logits