import sys
import os
sys.path.append(os.path.abspath("../embedding"))  # Adjust if needed

import embedding
from compute_loss import compute_losses
from diffusers import StableDiffusionPipeline
import torch
from matplotlib import pyplot as plt

def generate_images_from_embeddings_visualize(embedded_data_list, T=50, model_id="Nihirc/Prompt2MedImage"):
    """
    Generates and visualizes images using soft-combined embeddings across denoising steps.
    embedded_data_list: [[filename, img_embedding, neutral_desc_embedding, style_rich_desc_embedding]]
    """

    # Load the pre-trained diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cpu")

    all_generated_images = {}  # Store images per sample

    for sample in embedded_data_list:
        filename = sample[0]  # Get filename for reference
        lambda_t = embedding.get_lambda_schedule(T, mode="sigmoid")  
        c_t = embedding.soft_combine_embeddings(sample[2], sample[3], lambda_t)  # Soft combination

        generated_images = []  # Store images for this sample

        for t in range(T):  # Simulating diffusion steps
            with torch.no_grad():  # No gradients needed for inference
                img = pipe(prompt_embeds=c_t[t].unsqueeze(0), num_inference_steps=50).images[0]
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

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")

    for sample in embedded_data_list:
        lambda_t = embedding.get_lambda_schedule(T, mode="sigmoid")  # TODO: Experiment with schedules!
        c_t = embedding.soft_combine_embeddings(sample[2], sample[3], lambda_t)  # Soft combination
        
        # Generate an image using c_t at each step
        generated_images = []
        for t in range(T):  # Simulating diffusion steps
            img = pipe(prompt_embeds=c_t[t].unsqueeze(0))  # Ensure correct batch shape
            generated_images.append(img.images[0])  # Store generated image

        print(f"Generated images for {sample[0]}") 
        return generated_images  # Return list of images across diffusion steps

if __name__ == "__main__":
    """ load the pkl dataset and create embeddings for text and images """
    pickle_filename = 'C:\\Users\katha\OneDrive\Desktop\GNNS_project\code\GNNS_final_project\data\dataset.pkl'
    data_with_desc = embedding.load_data_add_descriptions(pickle_filename)
    data_w_embeddings = embedding.add_embeddings(data_with_desc)

    generate_images_from_embeddings_visualize(data_w_embeddings)

    