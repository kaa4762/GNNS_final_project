import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

def compute_losses(img_neutral, img_interpolated, img_stylized, text_neutral_emb, text_stylized_emb):
    """
    Computes CLIP loss + perceptual loss for disentanglement training.
    """
    # this loss makes no sense because we only have one image

    # Define transformation
    to_tensor = transforms.ToTensor()

    # Convert images to tensors if they are not already
    if not isinstance(img_neutral, torch.Tensor):
        img_neutral = to_tensor(img_neutral).to(DEVICE)

    if not isinstance(img_interpolated, torch.Tensor):
        img_interpolated = to_tensor(img_interpolated).to(DEVICE)

    if not isinstance(img_stylized, torch.Tensor):
        img_stylized = to_tensor(img_stylized).to(DEVICE)

    # Compute CLIP loss
    def clip_loss(img_emb_neutral, img_emb_interpolated, img_emb_stylized, text_neutral_emb, text_stylized_emb):
        direction_text = text_stylized_emb - text_neutral_emb
        direction_image = img_emb_stylized - img_emb_interpolated
        return -F.cosine_similarity(direction_text, direction_image, dim=-1).mean()
    
    # Compute perceptual loss (ensuring Xₜ remains semantically close to X₀)
    def perceptual_loss(img_neutral, img_interpolated):
        return F.l1_loss(img_neutral, img_interpolated)

    beta = 0.5  # Adjust as needed
    loss = clip_loss(img_neutral, img_interpolated, img_stylized, text_neutral_emb, text_stylized_emb) + beta * perceptual_loss(img_neutral, img_interpolated)
    
    return loss

def clip_loss(img_emb_interpolated, text_neutral_emb, text_stylized_emb, alpha=0.5):
        # Define transformation
    to_tensor = transforms.ToTensor()

    # Convert images to tensors if they are not already
    if not isinstance(text_neutral_emb, torch.Tensor):
        img_neutral = to_tensor(img_neutral).to(DEVICE)

    if not isinstance(img_emb_interpolated, torch.Tensor):
        img_interpolated = to_tensor(img_interpolated).to(DEVICE)

    if not isinstance(text_stylized_emb, torch.Tensor):
        img_stylized = to_tensor(img_stylized).to(DEVICE)
    sim_neutral = F.cosine_similarity(img_emb_interpolated, text_neutral_emb, dim=-1)
    sim_stylized = F.cosine_similarity(img_emb_interpolated, text_stylized_emb, dim=-1)

    # Encourage similarity to both, but prioritize stylized alignment
    return -((1 - alpha) * sim_neutral.mean() + alpha * sim_stylized.mean())

