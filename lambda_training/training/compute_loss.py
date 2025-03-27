import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import sys
import os
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
sys.path.append(os.path.abspath("../embedding"))  # Adjust if needed
from embedding.embedding import soft_combine_embeddings

def compute_losses(img_neutral, img_interpolated, img_stylized, text_neutral_emb, text_stylized_emb):
    """
    Computes CLIP loss + perceptual loss for disentanglement training, adapted from Wu et al. (2022).
    (This loss was not used for this training since we only have one image instead of one "neutral" and one "stylized", so it makes no sense here)
    """
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

def clip_loss(img_emb_interpolated, text_neutral_emb, text_stylized_emb, alpha=0.8):
    """
        Simple clip loss computation
    """
    sim_neutral = F.cosine_similarity(img_emb_interpolated, text_neutral_emb, dim=-1)
    sim_stylized = F.cosine_similarity(img_emb_interpolated, text_stylized_emb, dim=-1)

    # Encourage similarity to both, but prioritize stylized alignment
    return -((1 - alpha) * sim_neutral.mean() + alpha * sim_stylized.mean())


def clip_loss_alternative(desc_neutral, desc_stylized, lambda_t, T, num_inference_steps, pipe, clip_image_embedder, weight=0.8):
    """
        clip loss computation at every step T of the lambda schedule; image generation was moved from training func into this function
    """

    print(f"lambda_t grad_fn before soft combine before pipe: {lambda_t.grad_fn}")

    # Combine embeddings
    c_t, lambda_t = soft_combine_embeddings(lambda_t=lambda_t, c0=desc_neutral, c1=desc_stylized)

    print(f"lambda_t grad_fn after soft combine embeddings but before pipe: {lambda_t.grad_fn}")

    step_losses = []  # Store each step's loss

    # Loop over diffusion steps
    for t in range(T):
        empty_negative_prompt = torch.zeros_like(c_t[t]).to(DEVICE)

        output = pipe(
            prompt_embeds=c_t[t].unsqueeze(0),
            negative_prompt_embeds=empty_negative_prompt.unsqueeze(0),
            num_inference_steps=num_inference_steps
        )

        print(f"c_t grad_fn: {c_t.grad_fn}")

        if hasattr(output, 'latent_embeds'):
            img_emb = output.latent_embeds[0]
        else:
            img_emb = output.images[0]

        print("Before clip embedder forward: lambda_t.requires_grad =", lambda_t.requires_grad)
        print(f"lambda_t grad_fn: {lambda_t.grad_fn}")

        img_interpolated = clip_image_embedder.forward(img_emb).squeeze(0).to(DEVICE)

        print("After clip embedder forward: lambda_t.requires_grad =", lambda_t.requires_grad)
        print("After clip embedder forward: img_interpolated.requires_grad =", img_interpolated.requires_grad)
        print(f"img_interpolated grad_fn: {img_interpolated.grad_fn}")
        print(f"lambda_t grad_fn: {lambda_t.grad_fn}")

        # Compute individual losses
        similarity_loss = F.cosine_similarity(img_interpolated, desc_neutral)
        style_loss = F.cosine_similarity(img_interpolated, desc_stylized)

        step_loss = similarity_loss + style_loss

        # Apply weight and store loss
        weighted_loss = weight * step_loss
        step_losses.append(weighted_loss)

    # Sum all step losses **after** the loop
    total_loss = torch.sum(torch.stack(step_losses))

    # Add regularization term for lambda_t
    lambda_penalty = 0.01 * torch.sum(torch.nan_to_num(lambda_t, nan=0.0, posinf=1.0, neginf=-1.0) ** 2) # make sure we deal with NaN values accordingly
    total_loss = total_loss + lambda_penalty
    print(f"end of loss function lambda_t grad_fn: {lambda_t.grad_fn}")

    return total_loss, lambda_t
