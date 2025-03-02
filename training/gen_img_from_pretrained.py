import sys
import os
sys.path.append(os.path.abspath("../embedding"))  # Adjust if needed

import embedding
from compute_loss import compute_losses
from diffusers import StableDiffusionPipeline
import torch
from matplotlib import pyplot as plt

# Disable NSFW checker in pipeline since some of the chest xrays are accidentally flagged which returns a black image
def dummy_checker(images, **kwargs):
    return images, False  # Always return images without flagging them

def generate_images_from_embeddings_visualize(embedded_data_list, T=50, model_id="Nihirc/Prompt2MedImage"):
    """
    Generates and visualizes images using soft-combined embeddings across denoising steps.
    embedded_data_list: [[filename, img_embedding, neutral_desc_embedding, style_rich_desc_embedding]]
    """

    # Load the pre-trained diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cpu")
    pipe.safety_checker = dummy_checker

    all_generated_images = {}  # Store images per sample
    for sample in embedded_data_list:
        filename = sample[0]  # Get filename for reference
        lambda_t = embedding.get_lambda_schedule(T, mode="sigmoid")  
        c_t = embedding.soft_combine_embeddings(sample[2], sample[3], lambda_t)  # Soft combination

        generated_images = []  # Store images for this sample

        for t in range(T):  # Simulating diffusion steps
            with torch.no_grad():  # No gradients needed for inference
                empty_negative_prompt = torch.zeros_like(c_t[t]) # dummy because pipeline expects 2 embeddings

                img = pipe(prompt_embeds=c_t[t].unsqueeze(0), negative_prompt_embeds=empty_negative_prompt.unsqueeze(0), num_inference_steps=50).images[0]
                generated_images.append(img)  # Store generated image

        all_generated_images[filename] = generated_images  # Store all images

        # Visualizing the generated images
        plt.figure(figsize=(10, 2))
        for i in range(min(5, T)):  # Show up to 5 images for preview
            plt.subplot(1, 5, i + 1)
            plt.imshow(generated_images[i])
            plt.axis("off")
        plt.suptitle(f"Generated Images for {filename}")
        plt.show()

    return all_generated_images  # Return all images for further processing


def generate_images_from_embeddings(embedded_data_list, T=50, model_id="Nihirc/Prompt2MedImage"):
    """
    Generates images using soft-combined embeddings across denoising steps.
    embedded_data_list: [[filename, img_embedding, neutral_desc_embedding, style_rich_desc_embedding]]
    """

    # Load the pre-trained diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cpu")
    pipe.safety_checker = dummy_checker

    all_generated_images = {}  # Store images per sample
    for sample in embedded_data_list:
        filename = sample[0]  # Get filename for reference
        lambda_t = embedding.get_lambda_schedule(T, mode="sigmoid")  
        c_t = embedding.soft_combine_embeddings(sample[2], sample[3], lambda_t)  # Soft combination

        generated_images = []  # Store images for this sample

        for t in range(T):  # Simulating diffusion steps
            with torch.no_grad():  # No gradients needed for inference
                empty_negative_prompt = torch.zeros_like(c_t[t]) # dummy because pipeline expects 2 embeddings

                img = pipe(prompt_embeds=c_t[t].unsqueeze(0), negative_prompt_embeds=empty_negative_prompt.unsqueeze(0), num_inference_steps=50).images[0]
                generated_images.append(img)  # Store generated image

        all_generated_images[filename] = generated_images  # Store all images

        return generated_images  # Return list of images across diffusion steps


def train_lambda(embedded_data_list, T=50, epochs=10, lr=1e-3):
    """
    Optimizes lambda_t for better disentanglement.
    """
    lambda_t = embedding.get_lambda_schedule(T, mode="sigmoid").to("cuda").requires_grad_()  # Trainable λₜ

    optimizer = torch.optim.Adam([lambda_t], lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for sample in embedded_data_list:
            # Generate images using current λₜ
            generated_images = generate_images_from_embeddings([sample], T)
            img_neutral, img_stylized = sample[1], sample[3]  # Original neutral & style embeddings
            img_interpolated = generated_images[-1]  # Use last interpolated image

            # Compute loss
            loss = compute_losses(img_neutral, img_interpolated, img_stylized, sample[2], sample[3])

            # Backprop & update λₜ
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return lambda_t


if __name__ == "__main__":
    """ load the pkl dataset and create embeddings for text and images """
    pickle_filename = '\GNNS_final_project\data\dataset.pkl'
    data_with_desc = embedding.load_data_add_descriptions(pickle_filename)
    data_w_embeddings = embedding.add_embeddings(data_with_desc)

    generate_images_from_embeddings_visualize(data_w_embeddings)

    