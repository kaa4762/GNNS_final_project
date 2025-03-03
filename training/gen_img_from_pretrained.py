import sys
import os
sys.path.append(os.path.abspath("../embedding"))  # Adjust if needed

import embedding
from compute_loss import compute_losses
from diffusers import StableDiffusionPipeline
import torch
from matplotlib import pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
# Disable NSFW checker in pipeline since some of the chest xrays are accidentally flagged which returns a black image
def dummy_checker(images, **kwargs):
    return images, [False] * len(images)# Always return images without flagging them

def generate_images_from_embeddings_visualize(embedded_data_list, T=20, model_id="Nihirc/Prompt2MedImage"):
    """
    Generates and visualizes images using soft-combined embeddings across denoising steps.
    embedded_data_list: [[filename, img_embedding, neutral_desc_embedding, style_rich_desc_embedding]]
    """

    # Load the pre-trained diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(DEVICE)
    pipe.safety_checker = dummy_checker

    all_generated_images = {}  # Store images per sample
    for sample in embedded_data_list:
        filename = sample[0]  # Get filename for reference
        lambda_t = embedding.get_lambda_schedule(T, mode="sigmoid").to(DEVICE)
        c_t = embedding.soft_combine_embeddings(sample[2], sample[3], lambda_t).to(DEVICE)  # Soft combination

        generated_images = []  # Store images for this sample

        for t in range(T):  # Simulating diffusion steps
            with torch.no_grad():  # No gradients needed for inference
                empty_negative_prompt = torch.zeros_like(c_t[t]).to(DEVICE) # dummy because pipeline expects 2 embeddings

                img = pipe(prompt_embeds=c_t[t].unsqueeze(0), negative_prompt_embeds=empty_negative_prompt.unsqueeze(0), num_inference_steps=20).images[0]
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
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(DEVICE)
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

def plot_losses(train_losses, test_losses):
    """
    Plots training and testing loss curves over epochs.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o', color='b')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss", marker='s', color='r')
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Testing Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_lambda(embedded_data_list, T=50, epochs=10, lr=1e-3):
    """
    Optimizes lambda_t for better disentanglement.
    """
    lambda_t = embedding.get_lambda_schedule(T, mode="sigmoid").to(DEVICE).requires_grad_()  # Trainable λₜ

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

import random
#from sklearn.model_selection import train_test_split

def train_lambda_train_test(embedded_data_list, T=50, epochs=10, lr=1e-3, test_size=0.2):
    """
    Optimizes lambda_t for better disentanglement, with train-test split.
    """
    # Split data into training and testing sets
    train_data, test_data = None##train_test_split(embedded_data_list, test_size=test_size, random_state=42)

    # Trainable λₜ
    lambda_t = embedding.get_lambda_schedule(T, mode="sigmoid").to(DEVICE).requires_grad_()
    
    # Optimizer
    optimizer = torch.optim.Adam([lambda_t], lr=lr)

    # Store losses for plotting
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        total_train_loss = 0
        total_test_loss = 0

        # Training loop
        for sample in train_data:
            generated_images = generate_images_from_embeddings([sample], T)
            img_neutral, img_stylized = sample[1], sample[3]
            img_interpolated = generated_images[-1]

            loss = compute_losses(img_neutral, img_interpolated, img_stylized, sample[2], sample[3])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Testing loop (no gradient updates)
        with torch.no_grad():
            for sample in test_data:
                generated_images = generate_images_from_embeddings([sample], T)
                img_neutral, img_stylized = sample[1], sample[3]
                img_interpolated = generated_images[-1]

                loss = compute_losses(img_neutral, img_interpolated, img_stylized, sample[2], sample[3])
                total_test_loss += loss.item()

        # Store average losses per epoch
        avg_train_loss = total_train_loss / len(train_data)
        avg_test_loss = total_test_loss / len(test_data)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    return lambda_t, train_losses, test_losses

if __name__ == "__main__":
    """ load the pkl dataset and create embeddings for text and images """
    pickle_filename = 'C:\\Users\katha\OneDrive\Desktop\GNNS_project\code\GNNS_final_project\data\dataset.pkl'
    data_with_desc = embedding.load_data_add_descriptions(pickle_filename)
    data_w_embeddings = embedding.add_embeddings(data_with_desc)

    generate_images_from_embeddings_visualize(data_w_embeddings)

