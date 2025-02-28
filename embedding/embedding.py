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

""" 
    CLIP embedder classes adapted from https://github.com/UCSB-NLP-Chang/DiffusionDisentanglement/blob/main/ldm/modules/encoders/modules.py#L5
"""

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


""" 
    Create text descriptions 
"""
def summarize_labels(labels, max_items=3):
    """ Summarizes a long label list while keeping key findings. This is because there is a limit to the input of the CLIP text embedder 
    
    TODO: review if this is the best approach to deal with long descriptions """

    key_conditions = labels[:max_items]  # Take first 3 labels
    other_count = max(0, len(labels) - max_items)

    if other_count > 0:
        return f"{', '.join(key_conditions)}, and {other_count} other findings"
    else:
        return ", ".join(key_conditions)
    
def create_neutral_desc(sample):
    """Creates a neutral medical descriptor from a dataset entry
     sample: Shape [filename, img_array, orientation, labels] """
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
    sample: Shape [filename, img_array, orientation, labels]
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
        for sample in dataset[:10]: # just process 10 for faster testing
            neutral = create_neutral_desc(sample)
            style_rich = create_style_rich_desc(sample)
            imgs_w_desc.append([sample[0], sample[1], neutral, style_rich]) #filename, img, neutral desc, syle rich desc
    print("Descriptions added; first sample: ", imgs_w_desc[0])
    return imgs_w_desc

"""
    Create embeddings
"""
def add_embeddings(imgs_w_desc):
    """ turn the description strings and images into embeddings
    imgs_w_desc: Shape [[filename, img_array, neutral_desc, style_rich_desc]]"""
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

"""
    Soft combination of embeddings according to Disentanglement paper (Wu et al., 2022)
"""
def soft_combine_embeddings(c0, c1, lambda_t):
    """
    c0 (Tensor) The neutral text embedding. Shape: [batch, dim]
    c1 (Tensor): The style-rich text embedding. Shape: [batch, dim]
    lambda_t (Tensor): The combination weight (0 to 1). Shape: [T] or [T, 1]
    
    c_t (Tensor): The combined embedding over time. Shape: [T, batch, dim]
    """
    # Ensure lambda_t has correct shape for broadcasting
    lambda_t = lambda_t.view(-1, 1, 1)  # Shape: [T, 1, 1]

    # Linearly combine the embeddings over time
    c_t = lambda_t * c1 + (1 - lambda_t) * c0  # Shape: [T, batch, dim]
    
    return c_t

def get_lambda_schedule(T, mode="linear"):
    """
    Generates a lambda schedule over T timesteps
    """
    if mode == "linear":
        return torch.linspace(0, 1, steps=T)  # Linearly increasing
    elif mode == "sigmoid":
        x = torch.linspace(-6, 6, steps=T)  # Sigmoid range
        return torch.sigmoid(x)  # Smooth start and end
    elif mode == "cosine":
        return (1 - torch.cos(torch.linspace(0, 3.1416, steps=T))) / 2  # Cosine ease-in-out
    else:
        raise ValueError("Invalid mode! Choose 'linear', 'sigmoid', or 'cosine'.")


def test_soft_combined_embeddings(embedded_data_list, T=50):
    """
    embedded_data_list: Shape [[filename, img_embedding, neutral_desc_embedding, style_rich_desc_embedding]] 
    """
    for sample in embedded_data_list:
        lambda_t = get_lambda_schedule(T, mode="sigmoid")  # TODO: Try different schedules!

        # Compute the soft combination of embeddings
        c_t = soft_combine_embeddings(sample[2], sample[3], lambda_t)  
        #print(c_t)
        print(c_t.shape) #torch.Size([50, 1, 768])


if __name__ == "__main__":
    """ load the pkl dataset and create embeddings for text and images """
    pickle_filename = '\GNNS_final_project\data\dataset.pkl'
    data_with_desc = load_data_add_descriptions(pickle_filename)
    data_w_embeddings = add_embeddings(data_with_desc)
    test_soft_combined_embeddings(data_w_embeddings)

    