import torch
import torch.nn as nn
from transformers import WhisperModel
from peft import LoraConfig, get_peft_model


class WhisperAGNClassifier(nn.Module):
    def __init__(self, num_classes=2, use_lora=True, use_dora=False):
        super().__init__()

        whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
        self.encoder = whisper.encoder
        d_model = self.encoder.config.d_model  # 384 for tiny

        # ----- LoRA / DoRA -----
        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                use_dora=use_dora
            )
            self.encoder = get_peft_model(self.encoder, lora_config)

        # ----- Classifier head -----
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x: (B, 80, T) from WhisperFeatureExtractor
        """
        out = self.encoder(input_features=x).last_hidden_state
        pooled = out.mean(dim=1)
        return self.classifier(pooled)
