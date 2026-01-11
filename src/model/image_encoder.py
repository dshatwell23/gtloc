import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim=768, embedding_dim=512, freeze_backbone=True):
        super().__init__()

        self.backbone = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.freeze_backbone = freeze_backbone

        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.mlp = nn.Sequential(nn.Linear(768, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, embedding_dim))
    
    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.backbone.get_image_features(pixel_values=x)
        else:
            x = self.backbone.get_image_features(pixel_values=x)
        # x = self.backbone.get_image_features(pixel_values=x)
        x = self.mlp(x)
        return x
    
    def forward_image(self, image):
        x = self.backbone.get_image_features(pixel_values=image)
        x = self.mlp(x)
        return x