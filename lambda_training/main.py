import sys
import os
sys.path.append(os.path.abspath("./embedding"))  # Adjust if needed
sys.path.append(os.path.abspath("./training"))  # Adjust if needed
from embedding.embedding import load_data_add_descriptions, get_lambda_schedule, add_embeddings_single_img, add_embeddings
from training.gen_img_from_pretrained import train_lambda_train_test_alternative, generate_images_from_embeddings_visualize, load_lambda_model
from diffusers import StableDiffusionPipeline
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_PATH = './data/dataset_small.pkl' 

def visualize_lambda_schedules(model):
    """
        generate imgs with the different lambda schedules
    """
    data_with_desc = load_data_add_descriptions(DATASET_PATH)
    data_w_embeddings = add_embeddings_single_img(data_with_desc, model=model)
    print("Lambda schedule: linear")
    lambda_t = get_lambda_schedule(T=20, mode="linear").to(DEVICE)
    generate_images_from_embeddings_visualize(embedded_data_list=data_w_embeddings[:3],  lambda_t=lambda_t, num_inference_steps=50, pipe=model)
    print("Lambda schedule: sigmoid")
    lambda_t = get_lambda_schedule(T=20, mode="sigmoid").to(DEVICE)
    generate_images_from_embeddings_visualize(embedded_data_list=data_w_embeddings[:3], lambda_t=lambda_t, num_inference_steps=50, pipe=model)
    print("Lambda schedule: cosine")
    lambda_t = get_lambda_schedule(T=20, mode="cosine").to(DEVICE)
    generate_images_from_embeddings_visualize(embedded_data_list=data_w_embeddings[:3], lambda_t=lambda_t, num_inference_steps=50, pipe=model)

def train(model):
    """
        load dataset and call training function from gen_imgs_from_pretrained.py
    """
    batch_size = 2
    num_inference_steps = 20
    data_with_desc = load_data_add_descriptions(DATASET_PATH)
    data_w_embeddings = add_embeddings(data_with_desc, batch_size=batch_size)
    T=6
    lambda_t = torch.nn.Parameter(get_lambda_schedule(T=T, mode="sigmoid").to(dtype=torch.float16, device=DEVICE), requires_grad=True)
    return train_lambda_train_test_alternative(
        data_w_embeddings,
        T,
        lambda_t=lambda_t,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        save_path="./",
        pipe=model)

def train_continue(model):
    """
        continue with a loaded lambda_t from the .pth model
    """
    loaded_lambda_t, optimizer = load_lambda_model(load_path="./")
    loaded_lambda_t = loaded_lambda_t.squeeze(dim=1).squeeze(dim=1)  # Removes both extra dimensions

    print(loaded_lambda_t.shape)
    batch_size = 2
    num_inference_steps = 20
    data_with_desc = load_data_add_descriptions(DATASET_PATH)
    data_w_embeddings = add_embeddings(data_with_desc, batch_size=batch_size)
    T=6
    lambda_values = loaded_lambda_t.clone().detach().cpu().numpy().tolist()  # Convert tensor to list
    print(lambda_values)
    lambda_t = torch.nn.Parameter(loaded_lambda_t.to(dtype=torch.float16, device=DEVICE), requires_grad=True)
    print(lambda_t.shape)

    return train_lambda_train_test_alternative(
        data_w_embeddings,
        T,
        lambda_t=lambda_t,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        save_path="./",
        pipe=model
    )

def inference(model):
    """
        compare lambda image generations before and after training
    """
    loaded_lambda_t, optimizer = load_lambda_model(load_path="./")
    loaded_lambda_t = loaded_lambda_t.squeeze(dim=1).squeeze(dim=1)  # Removes both extra dimensions
    batch_size = 1
    num_inference_steps = 20
    data_with_desc = load_data_add_descriptions(DATASET_PATH)
    data_w_embeddings = add_embeddings(data_with_desc, batch_size=batch_size)

    print("Lambda schedule: sigmoid")
    lambda_t = get_lambda_schedule(T=6, mode="sigmoid").to(DEVICE)
    generate_images_from_embeddings_visualize(embedded_data_list=data_w_embeddings[:1], T=6, lambda_t=lambda_t, num_inference_steps=num_inference_steps, pipe=model)

    print("Lambda schedule after 10 epochs")
    generate_images_from_embeddings_visualize(embedded_data_list=data_w_embeddings[:1], T=6, lambda_t=loaded_lambda_t, num_inference_steps=num_inference_steps, pipe=model)

if __name__ == '__main__':
    model = StableDiffusionPipeline.from_pretrained("Nihirc/Prompt2MedImage")
    #visualize_lambda_schedules(model)
    #train(model)
    #train_continue(model)
    inference(model)