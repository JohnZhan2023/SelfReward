'''
A preference model based on the CLIP model.
The frozen CLIP model will process the input image and then the embeddings will go through the linear layer to get the two classes' scores.
'''
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPProcessor

class CLIPForBinaryClassification(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", num_labels=2):
        super(CLIPForBinaryClassification, self).__init__()
        # only use the vision model
        self.clip_vision = CLIPVisionModel.from_pretrained(model_name)
        # freeze the vision model
        for param in self.clip_vision.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(self.clip_vision.config.hidden_size, 256), # output of the vision model
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, pixel_values=None):
        # get the vision embeddings
        outputs = self.clip_vision(pixel_values=pixel_values)
        image_embeds = outputs.last_hidden_state[:, 0, :]  # get the CLS token
        logits = self.classifier(image_embeds)
        return logits