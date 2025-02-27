import torch
import torch.nn as nn
import clip
import numpy as np
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import AutoTokenizer, AutoModel
import kornia
import zipfile
import pickle

class FrozenClipTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cpu", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """Uses the CLIP image encoder."""
    def __init__(self, model="ViT-L/14", device='cpu', antialias=False):
        super().__init__()
        self.model, _ = clip.load(model, device=device)
        self.device = device
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def forward(self, x):
        """Encodes the image into embeddings."""
        x = self.preprocess(x)
        return self.model.encode_image(x)

    def encode(self, x):
        """Encodes an image into CLIP embedding."""
        return self(x)
    
    def preprocess(self, x):
        """ Resize and normalize image for CLIP """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)  # Convert NumPy to Tensor

        if x.ndim == 2:  # Grayscale images (H, W)
            x = x.unsqueeze(0)  # Add channel dimension → (1, H, W)

        if x.shape[0] == 1:  # Convert grayscale to RGB by repeating channels #TODO save them as RGB in dataset?
            x = x.repeat(3, 1, 1)  # Now shape is (3, H, W)

        x = x / 255.0  # Normalize pixel values to [0,1]

        x = kornia.geometry.resize(x.unsqueeze(0), (224, 224),
                                interpolation='bicubic', align_corners=True,
                                antialias=self.antialias)  # Add batch dim

        x = (x - self.mean[:, None, None]) / self.std[:, None, None]  # Normalize
        return x

def summarize_labels(labels, max_items=3):
    """ Summarizes a long label list while keeping key findings. This is because there is a limit to the input of the CLIP text emnbedder """
    key_conditions = labels[:max_items]  # Take first 3 labels
    other_count = max(0, len(labels) - max_items)

    if other_count > 0:
        return f"{', '.join(key_conditions)}, and {other_count} other findings"
    else:
        return ", ".join(key_conditions)
    
def create_neutral_desc(sample):
    """Creates a neutral medical descriptor from a dataset entry
     sample: (filename, img_array, orientation, labels)"""
    filename, img_array, orientation, labels = sample
    
    # Convert label string to a proper list
    labels = eval(labels) if isinstance(labels, str) else labels  
    # Join labels into a sentence
    label_text = summarize_labels(labels)  # Summarize findings
    if label_text == "normal":
        return f"An X-ray of a patient with no findings"
    return f"An X-ray of a patient with {label_text}."

def create_style_rich_desc(sample):
    """
    Creates a style-rich descriptor with more context
    sample: (filename, img_array, orientation, labels)
    """
    filename, img_array, orientation, labels = sample
    
    # Convert label string to a proper list
    labels = eval(labels) if isinstance(labels, str) else labels  
    # Join labels into a sentence
    label_text = summarize_labels(labels)  # Summarize findings
    if label_text == "normal":
        return f"An X-ray of a patient with no findings taken in {orientation} orientation."
    return f"An X-ray of a patient with {label_text}, taken in {orientation} orientation."

def load_data_add_descriptions(pickle_filename):
    """ Adds description strings to the dataset which are later turned into embeddings """
    with open(pickle_filename, "rb") as f:
        dataset = pickle.load(f)
        imgs_w_desc = list()
        for sample in dataset:
            neutral = create_neutral_desc(sample)
            style_rich = create_style_rich_desc(sample)
            imgs_w_desc.append([sample[0], sample[1], neutral, style_rich]) #filename, img, neutral desc, syle rich desc
    print("Descriptions added; first sample: ", imgs_w_desc[0])
    return imgs_w_desc

def add_embeddings(imgs_w_desc):
    """ turn the description strings and images into embeddings
    imgs_w_desc: (filename, img_array, neutral_desc, style_rich_desc)"""
    clip_image_embedder = FrozenClipImageEmbedder()
    clip_text_embedder = FrozenClipTextEmbedder()
    # Freeze models for inference
    clip_image_embedder.eval()
    clip_text_embedder.eval()

    embedded_data = []
    
    with torch.no_grad():  # No gradients needed
        for sample in imgs_w_desc:
            filename, img_array, neutral_desc, style_rich_desc = sample

            # Convert text to embeddings
            neutral_embedding = clip_text_embedder.encode([neutral_desc]).squeeze(1)  # Shape: (1, D) → (D,)
            style_embedding = clip_text_embedder.encode([style_rich_desc]).squeeze(1)

            # Convert image to embedding
            img_embedding = clip_image_embedder.forward(img_array).squeeze(0)  # Remove batch dim

            # Store results
            embedded_data.append((filename, img_embedding.cpu(), neutral_embedding.cpu(), style_embedding.cpu()))
    print("Embeddings added; first sample: ", embedded_data[0])
    return embedded_data


if __name__ == "__main__":
    """ load the pkl dataset and create embeddings for text and images """
    pickle_filename = "./data/dataset.pkl"
    data_with_desc = load_data_add_descriptions(pickle_filename)
    data_w_embeddings = add_embeddings(data_with_desc)

    