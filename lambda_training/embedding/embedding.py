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
import torchvision.transforms as transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
    CLIP embedder classes adapted from https://github.com/UCSB-NLP-Chang/DiffusionDisentanglement/blob/main/ldm/modules/encoders/modules.py#L5
"""
class FrozenClipTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device=DEVICE, max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device=DEVICE) #CLIPModel.from_pretrained(med_clip_model).to(DEVICE)#
        self.device = DEVICE
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
    def __init__(self, model='ViT-L/14', device=DEVICE, antialias=False):
        super().__init__()
        self.model, _ = clip.load(model, device=DEVICE)
        self.device = DEVICE
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def forward(self, x):
        """Encodes the image into embeddings."""
        x = self.preprocess(x).to(self.device)
        return self.model.encode_image(x).to(self.device)

    def encode(self, x):
        """Encodes an image into CLIP embedding."""
        return self(x)

    def preprocess(self, x):
            """ Resize and normalize image for CLIP """
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)  # Convert NumPy to Tensor
            if isinstance(x, Image.Image):
                x = transforms.ToTensor()(x)  # Converts to (3, H, W), normalized to [0,1]
            #print('ndims:', x.ndim, ' shape:' , x.shape)
            if x.ndim == 2:  # Grayscale images (H, W)
                x = x.unsqueeze(0)  # Add channel dimension → (1, H, W)
                x = x.repeat(3, 1, 1)  # Convert to 3 channels → (3, H, W)

            if x.shape[0] == 1:  # Convert grayscale to RGB by repeating channels
                x = x.repeat(3, 1, 1)  # Now shape is (3, H, W)

            x = x / 255.0  # Normalize pixel values to [0,1]

            # Make sure x is on the same device as self.mean
            x = x.to(self.mean.device)

            x = kornia.geometry.resize(x.unsqueeze(0), (224, 224),
                                        interpolation='bicubic', align_corners=True,
                                        antialias=self.antialias)  # Add batch dim

            # Move self.mean and self.std to the same device as x
            x = (x - self.mean.to(x.device)[:, None, None]) / self.std.to(x.device)[:, None, None]

            return x

"""
    CLIP embedder classes of the stable diffusion model itself
"""

class FrozenSDTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, model, device=DEVICE, max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model = model.text_encoder #StableDiffusionPipeline.from_pretrained("Nihirc/Prompt2MedImage").text_encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")  # Use CLIP tokenizer #self.tokenizer = CLIPTokenizer.from_pretrained("Nihirc/Prompt2MedImage")
        self.device = DEVICE
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, text):
        # Tokenize input text
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move tokens to the correct device
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # Make sure the model itself is also on the same device!
        self.model = self.model.to(self.device)

        # Pass through the text encoder
        with torch.no_grad():
            text_embeddings = self.model(input_ids=tokens["input_ids"])[0]  # No need to call `.to(self.device)` again

        # Normalize embeddings if needed
        if self.normalize:
            text_embeddings = text_embeddings / torch.linalg.norm(text_embeddings, dim=-1, keepdim=True)

        return text_embeddings

    def encode(self, text):
        text_embeddings = self.forward(text).to(self.device)
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings[:, None, :]
        text_embeddings = text_embeddings.repeat(1, self.n_repeat, 1).to(self.device)
        return text_embeddings


class FrozenSDImageEmbedder(nn.Module):
    """Uses the CLIP image encoder."""
    def __init__(self, model, device=DEVICE, antialias=False):
        super().__init__()

        self.device = device
        self.antialias = antialias
        self.model = model

        # Access the VAE model from the pipeline and move to device
        self.vae = self.model.vae.to(self.device)  # Get the VAE model, move to device
        self.register_buffer('mean', torch.Tensor([0.5, 0.5, 0.5]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.5, 0.5, 0.5]), persistent=False)

    def forward(self, x):
        """Encodes the image into embeddings using the diffusion model's image encoder."""
        x = self.preprocess(x).to(self.device)  # Ensure input is on the correct device
        with torch.no_grad():
            # Use VAE's encoding method to get latent representation
            encoded = self.vae.encode(x).latent_dist.sample()
            encoded = encoded * 0.18215  # Apply scaling factor for diffusion
        return encoded.to(self.device)  # Ensure output is on the correct device

    def preprocess(self, x):
        """Resize and normalize image for the diffusion model."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(x, Image.Image):
            x = transforms.ToTensor()(x)

        if x.ndim == 2:  # If grayscale, convert to 3-channel RGB
            x = x.unsqueeze(0).repeat(3, 1, 1)

        if x.shape[0] == 1:  # Single-channel grayscale
            x = x.repeat(3, 1, 1)

        x = x.unsqueeze(0)  # Add batch dimension

        # Resize to 224x224 (standard for many models)
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)

        # Normalize to [-1, 1] for diffusion models
        x = (x - 0.5) / 0.5

        return x.to(self.device)  # Move to the correct device

"""
    Create text descriptions
"""
def summarize_labels(labels, max_items=1):
    """ Summarizes a long label list while keeping key findings. This is because there is a limit to the input of the CLIP text embedder

    TODO: review if this is the best approach to deal with long descriptions """

    key_conditions = labels[:max_items]  # Take first labels
    other_count = max(0, len(labels) - max_items)
    # maybe try simpler prompts with just one condition
    #if other_count > 0:
    #    return f"{', '.join(key_conditions)}, and {other_count} other findings"
    #else:
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
        return f"A chest X-ray with no findings."
    if label_text == "unchanged":
        return f"A chest X-ray with unchanged findings."
    return f"A chest X-ray with {label_text}."

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
        return f"A chest X-ray with no findings taken in {orientation} orientation."
    if label_text == "unchanged":
        return f"A chest X-ray with unchanged findings, taken in {orientation} orientation."
    return f"A chest X-ray with {label_text}, taken in {orientation} orientation."

def load_data_add_descriptions(pickle_filename):
    """ Adds description strings to the dataset which are later turned into embeddings """
    with open(pickle_filename, "rb") as f:
        dataset = pickle.load(f)
        imgs_w_desc = list()
        for sample in dataset[:10]: #small subset to test training
            neutral = create_neutral_desc(sample)
            style_rich = create_style_rich_desc(sample)
            imgs_w_desc.append([sample[0], sample[1], neutral, style_rich]) #filename, img, neutral desc, syle rich desc

    for sample in imgs_w_desc[:10]:
        print(sample[0], sample[2], sample[3])
    print('num of samples: ', len(imgs_w_desc))
    return imgs_w_desc

"""
    Create embeddings
"""

def add_embeddings_single_img(imgs_w_desc, model=None):
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
            neutral_embedding = clip_text_embedder.encode([neutral_desc]).squeeze(1).to(DEVICE)  # Shape: (1, D) → (D,)
            style_embedding = clip_text_embedder.encode([style_rich_desc]).squeeze(1).to(DEVICE)

            # Convert image to embedding
            img_embedding = clip_image_embedder.forward(img_array).squeeze(0).to(DEVICE)
            # Store results
            embedded_data.append((filename, img_embedding, neutral_embedding, style_embedding))
    #print("Embeddings added; first sample: ", embedded_data[0])
    return embedded_data


def add_embeddings(imgs_w_desc, batch_size):
    """ turn the description strings and images into embeddings in batches
    imgs_w_desc: Shape [[filename, img_array, neutral_desc, style_rich_desc]]
    """
    clip_image_embedder = FrozenClipImageEmbedder()
    clip_text_embedder = FrozenClipTextEmbedder()

    # Freeze models for inference
    clip_image_embedder.eval()
    clip_text_embedder.eval()

    embedded_data = []

    with torch.no_grad():  # No gradients needed
        for i in range(0, len(imgs_w_desc), batch_size):
            batch_samples = imgs_w_desc[i:i + batch_size]
            filenames = [sample[0] for sample in batch_samples]
            images = [sample[1] for sample in batch_samples]
            neutral_descs = [sample[2] for sample in batch_samples]
            style_descs = [sample[3] for sample in batch_samples]

            # Process the batch
            neutral_embeddings =  []
            for neutral_desc in neutral_descs:
              #[print(neutral_desc)
              neutral_embedding = clip_text_embedder.encode(neutral_desc).to(DEVICE)
              neutral_embeddings.append(neutral_embedding)
            style_embeddings = []
            for style_desc in style_descs:
              #print(style_desc)
              style_embedding = clip_text_embedder.encode(style_desc).to(DEVICE)
              style_embeddings.append(style_embedding)
            img_embeddings = []
            for image in images:
              img_embedding = clip_image_embedder.forward(image).squeeze(0).to(DEVICE)
              img_embeddings.append(img_embedding)
            for idx, sample in enumerate(batch_samples):
              #print(sample)
              #print(len(img_embeddings), len(neutral_embeddings), len(style_embeddings))
              embedded_data.append([filenames[idx], img_embeddings[idx], neutral_embeddings[idx], style_embeddings[idx]])
              #print(embedded_data[idx])

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
    lambda_t = lambda_t.reshape(-1, 1, 1)  # Safe way to reshape without breaking the computation graph

    # Linearly combine the embeddings over time
    c_t = lambda_t * c1 + (1 - lambda_t) * c0  # Shape: [T, batch, dim]

    return c_t, lambda_t

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
        c_t, _ = soft_combine_embeddings(sample[2], sample[3], lambda_t)
        #print(c_t)


if __name__ == "__main__":
    """ load the pkl dataset and create embeddings for text and images """
    pickle_filename = '\GNNS_final_project\data\dataset.pkl'
    data_with_desc = load_data_add_descriptions(pickle_filename)
    data_w_embeddings = add_embeddings(data_with_desc)
    test_soft_combined_embeddings(data_w_embeddings)

    