import sys
import os
sys.path.append(os.path.abspath("../embedding"))  # Adjust if needed
sys.path.append(os.path.abspath("./"))  # Adjust if needed
from embedding.embedding import soft_combine_embeddings, FrozenClipImageEmbedder
import training.compute_loss as compute_loss
import csv
from diffusers import StableDiffusionPipeline
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Disable NSFW checker in pipeline since some of the chest xrays are accidentally flagged which returns a black image
def dummy_checker(images, **kwargs):
    return images, [False] * len(images)# Always return images without flagging them

def generate_images_from_embeddings_visualize(embedded_data_list, lambda_t, pipe, T=20, num_inference_steps=50):
    """
    Generates and visualizes images using soft-combined embeddings across denoising steps.
    embedded_data_list: [[filename, img_embedding, neutral_desc_embedding, style_rich_desc_embedding]]
    """

    # Load the pre-trained diffusion model
    pipe = pipe.to(DEVICE) #StableDiffusionPipeline.from_pretrained(model_id).to(DEVICE)
    pipe.safety_checker = dummy_checker

    all_generated_images = {}  # Store images per sample
    for sample in embedded_data_list:
        filename = sample[0]  # Get filename for reference
        c_t, lambda_t = soft_combine_embeddings(sample[2], sample[3], lambda_t)  # Soft combination

        generated_images = []  # Store images for this sample

        for t in range(T):  # Simulating diffusion steps
                empty_negative_prompt = torch.zeros_like(c_t[t]).to(DEVICE) # dummy because pipeline expects 2 embeddings

                img = pipe(prompt_embeds=c_t[t].unsqueeze(0), negative_prompt_embeds=empty_negative_prompt.unsqueeze(0), num_inference_steps=num_inference_steps).images[0]
                generated_images.append(img)  # Store generated image

        all_generated_images[filename] = generated_images  # Store all images

        # Visualizing the generated images
        plt.figure(figsize=(10, 2))
        for i in range(min(6, T)):  # Show up to 6 images for preview
            plt.subplot(1, 6, i + 1)
            plt.imshow(generated_images[i])
            plt.axis("off")
        plt.suptitle(f"Generated Images for {filename}")
        plt.show()

    return all_generated_images  # Return all images for further processing


def generate_images_from_embeddings(embedded_data_list, T, lambda_t, num_inference_steps, pipe):
    """
    Generates images using soft-combined embeddings across denoising steps.
    embedded_data_list: [[filename, img_embedding, neutral_desc_embedding, style_rich_desc_embedding]]
    """

    # Load the pre-trained diffusion model
    pipe = pipe.to(DEVICE) #StableDiffusionPipeline.from_pretrained(model_id).to(DEVICE)
    pipe.safety_checker = dummy_checker

    all_generated_images = {}  # Store images per sample
    for sample in embedded_data_list:
        filename = sample[0]  # Get filename for reference
        print("generate images: processing file ", filename)
        c_t, lambda_t = soft_combine_embeddings(sample[2], sample[3], lambda_t)  # Soft combination

        generated_images = []  # Store images for this sample

        for t in range(T):  # Simulating diffusion steps
                empty_negative_prompt = torch.zeros_like(c_t[t]).to(DEVICE) # dummy because pipeline expects 2 embeddings
                output = pipe(prompt_embeds=c_t[t].unsqueeze(0), negative_prompt_embeds=empty_negative_prompt.unsqueeze(0), num_inference_steps=num_inference_steps)
                if hasattr(output, 'latent_embeds'):  # Check if embeddings are available!
                    img_emb = output.latent_embeds[0]
                else:
                    img_emb = output.images[0]  # Fallback to image
                generated_images.append(img_emb)  # Store generated image

        all_generated_images[filename] = generated_images  # Store all images
        print("end of generate function: lambda_t.requires_grad =", lambda_t.requires_grad)
        all_generated_images[filename] = generated_images  # Store all images

        return generated_images  # Return list of images across diffusion steps


def train_lambda_train_test_old(embedded_data_list, T, save_path, lambda_t, num_inference_steps, pipe, loaded_optimizer=None, epochs=10, lr=1e-3, batch_size=8):
    """
    Training function for optimizing lambda. This function wasn't used for training since there were issues with lambda in the computation graph (the lambda gradients
    at some points got lost and the optimizer didn't update anything). Here, also the old simple clip_loss was used that only computes the clip loss at the end of the image generation.
    In order to get the training to work i made some changes to the loss function (see clip_loss_alternative) and used the alternative training function below instead.
    Still, I left it here for the sake of completeness.
    """
    # Logging setup
    log_file = "../logs/training_log.csv"
    lambda_log_file = "../logs/lambda_log.csv"
    lambda_values = lambda_t.detach().cpu().numpy().tolist()  # Convert tensor to list

    # Create or overwrite lambda log file
    if not os.path.exists(lambda_log_file):
      with open(lambda_log_file, mode="w", newline="") as file:
          writer = csv.writer(file)
          header = [f"Lambda_{t+1}" for t in range(T)]  # Column names
          writer.writerow(header)

    # Append first lambda to CSV
    #with open(lambda_log_file, mode="a", newline="") as file:
    #      writer = csv.writer(file)
    #      writer.writerow(lambda_values)  # Write λ values

    # Check if the file exists, if not create it with headers
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Lambda", "Train Loss", "Test Loss"])


    # Split data into training and testing sets
    clip_image_embedder = FrozenClipImageEmbedder()

    # Optimizer
    if loaded_optimizer is None:
      optimizer = torch.optim.Adam([lambda_t], lr=lr)
    else:
      optimizer = loaded_optimizer
      print("loaded optimizer checkpoint")

    # Store losses for plotting
    train_losses = []
    test_losses = []
    print("Start training loop")

    # Split the embedded data into train and test sets
    num_samples = len(embedded_data_list)
    train_size = int(0.8 * num_samples)  # 80% for training
    test_size = num_samples - train_size  # 20% for testing
    train_data = embedded_data_list[:train_size]
    test_data = embedded_data_list[train_size:]

    # Define a transformation to convert PIL image -> Tensor
    transform = transforms.Compose([
         transforms.ToTensor(),  # Converts PIL image to tensor
          transforms.Resize((224, 224)),  # Resize to CLIP input size
          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
            ])
    # Train loop with batch processing
    for epoch in range(epochs):
        total_train_loss = 0
        total_test_loss = 0

        # Batching for training
        for i in range(0, len(train_data), batch_size):
            batch_samples = train_data[i:i + batch_size]
            filenames = [sample[0] for sample in batch_samples]
            neutral_desc_embeddings = [sample[2] for sample in batch_samples]
            stylized_desc_embeddings = [sample[3] for sample in batch_samples]

            # Generate images for each sample in the batch (individual call to generate_images_from_embeddings)
            generated_images = []
            print("Train: generate images")
            for sample in batch_samples:
                generated_images_for_sample = generate_images_from_embeddings([sample], T=T, lambda_t=lambda_t, num_inference_steps=num_inference_steps, pipe=pipe)
                img_interpolated = generated_images_for_sample[-1]
                img_interpolated = transform(img_interpolated).to(DEVICE)

                generated_images.append(img_interpolated)

            # Calculate loss for the batch
            batch_loss = 0
            print("Train: calculate loss")
            for idx, sample in enumerate(batch_samples):
                img_interpolated = generated_images[idx]
                desc_neutral, desc_stylized = sample[2], sample[3]

                #img_interpolated = transforms.ToTensor()(img_interpolated).to_device(DEVICE)#unsqueeze(0).to(DEVICE)
                img_interpolated = clip_image_embedder.forward(img_interpolated).squeeze(0).to(DEVICE)

                # Compute loss for this sample
                loss = compute_loss.clip_loss(img_interpolated, desc_neutral, desc_stylized)
                batch_loss += loss

            # Backpropagate
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_train_loss += batch_loss.item()

        # Testing loop (Batch processing for testing)
        for i in range(0, len(test_data), batch_size):
            batch_samples = test_data[i:i + batch_size]
            filenames = [sample[0] for sample in batch_samples]
            neutral_desc_embeddings = [sample[2] for sample in batch_samples]
            stylized_desc_embeddings = [sample[3] for sample in batch_samples]

            # Generate images for each sample in the batch (individual call to generate_images_from_embeddings)
            generated_images = []
            print("Test: generate images")
            for sample in batch_samples:
                # Generate images for each sample as done previously
                generated_images_for_sample = generate_images_from_embeddings([sample], T=T, lambda_t=lambda_t, num_inference_steps=num_inference_steps, pipe=pipe)
                img_interpolated = generated_images_for_sample[-1]
                img_interpolated = transform(img_interpolated).to(DEVICE)
                generated_images.append(img_interpolated)

            # Calculate loss for the batch
            batch_loss = 0
            print("Test: calculate loss")
            for idx, sample in enumerate(batch_samples):
                img_interpolated = generated_images[idx]
                desc_neutral, desc_stylized = sample[2], sample[3]

                #img_interpolated = transforms.ToTensor()(img_interpolated).to(DEVICE)#unsqueeze(0).to(DEVICE)
                img_interpolated = clip_image_embedder.forward(img_interpolated).squeeze(0).to(DEVICE)

                # Compute loss for this sample
                loss = compute_loss.clip_loss(img_interpolated, desc_neutral, desc_stylized)
                batch_loss += loss.item()

            total_test_loss += batch_loss

        # Print training and testing losses
        avg_train_loss = total_train_loss / len(train_data)
        avg_test_loss = total_test_loss / len(test_data)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        # Store average losses per epoch
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # Save model and losses
        save_dict = {
            "lambda_t": lambda_t.detach().cpu(),  # Move to CPU before saving
            "optimizer_state": optimizer.state_dict(),
            "train_losses": train_losses,
            "test_losses": test_losses
        }
        torch.save(save_dict, save_path)
        print(f"Model saved at: {save_path}")

        # Open in append mode and log data
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([lambda_t.mean().item(), avg_train_loss, avg_test_loss])  # Logging the mean of lambda_t

        lambda_values = lambda_t.detach().cpu().numpy().tolist()  # Convert tensor to list

        # Append lambda to CSV
        with open(lambda_log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(lambda_values)  # Write λ values
            print(f"Epoch logged to {log_file} and {lambda_log_file}.")

    return lambda_t, train_losses, test_losses

# Disable NSFW checker in pipeline since some of the chest xrays are accidentally flagged which returns a black image
def dummy_checker(images, **kwargs):
    return images, [False] * len(images)# Always return images without flagging them

def train_lambda_train_test_alternative(embedded_data_list, T, save_path, lambda_t, num_inference_steps, pipe, loaded_optimizer=None, epochs=10, lr=1e-3, batch_size=8):
    """ 
    An alternative training function that uses clip_loss_alternative for loss computation. Here (unlike the old training function) lambda 
    actually successfully updates over the epochs. Losses and lambdas are logged as csv files.
    """

    # Logging setup
    log_file = "../logs/training_log_small.csv"
    lambda_log_file = "../logs/lambda_log_small.csv"

    # Create or overwrite lambda log file
    if not os.path.exists(lambda_log_file):
      with open(lambda_log_file, mode="w", newline="") as file:
          writer = csv.writer(file)
          header = [f"Lambda_{t+1}" for t in range(T)]  # Column names
          writer.writerow(header)
    lambda_values = lambda_t.clone().detach().cpu().numpy().tolist()  # Convert tensor to list

    # Append first lambda to CSV
    with open(lambda_log_file, mode="a", newline="") as file:
          writer = csv.writer(file)
          writer.writerow(lambda_values)  # Write λ values

    # Check if the file exists, if not create it with headers
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Train Loss", "Test Loss"])

    print("Is lambda_t a Parameter?", isinstance(lambda_t, torch.nn.Parameter))

    # Split data into training and testing sets
    clip_image_embedder = FrozenClipImageEmbedder()

    # Optimizer
    if loaded_optimizer is None:
      optimizer = torch.optim.Adam([lambda_t], lr=lr)
    else:
      optimizer = loaded_optimizer
      print("loaded optimizer checkpoint")

    print("Lambda_t in optimizer:", any(p is lambda_t for p in optimizer.param_groups[0]['params']))
    for group in optimizer.param_groups:
        for p in group['params']:
            print(f"Optimizer param: {p.shape}, requires_grad={p.requires_grad}")
    # Store losses for plotting
    train_losses = []
    test_losses = []
    lambdas = []
    print("Start training loop")

    # Split the embedded data into train and test sets
    num_samples = len(embedded_data_list)
    train_size = int(0.8 * num_samples)  # 80% for training
    test_size = num_samples - train_size  # 20% for testing

    # Split the embedded data into train and test sets
    train_data = embedded_data_list[:train_size]
    test_data = embedded_data_list[train_size:]

    # Define a transformation to convert PIL image -> Tensor
    transform = transforms.Compose([
         transforms.ToTensor(),  # Converts PIL image to tensor
          transforms.Resize((224, 224)),  # Resize to CLIP input size
          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
            ])

    pipe.safety_checker = dummy_checker # prevent imgs from being flagged as NSFW

    # Train loop with batch processing
    for epoch in range(epochs):
        print(f"--------------EPOCH {epoch} START--------------")
        total_train_loss = 0
        total_test_loss = 0
        # Batching for training
        for i in range(0, len(train_data), batch_size):
            batch_samples = train_data[i:i + batch_size]
            filenames = [sample[0] for sample in batch_samples]
            neutral_desc_embeddings = [sample[2] for sample in batch_samples]
            stylized_desc_embeddings = [sample[3] for sample in batch_samples]
            # Calculate loss for the batch
            batch_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            print("Train: calculate loss")
            for idx, sample in enumerate(batch_samples):
                desc_neutral, desc_stylized = sample[2].requires_grad_(), sample[3].requires_grad_()
                original_lambda_shape = lambda_t.shape
                total_loss, lambda_t = compute_loss.clip_loss_alternative(desc_neutral, desc_stylized, lambda_t, T, num_inference_steps, pipe, clip_image_embedder, weight=0.8)
                print(f"back in training func lambda_t grad_fn: {lambda_t.grad_fn}")
                lambda_t.reshape(original_lambda_shape)
                lambda_t.retain_grad()
                print(f"back in training func after reshape lambda_t grad_fn: {lambda_t.grad_fn}")


                batch_loss = batch_loss + total_loss
                print(f"batch_loss.requires_grad = {batch_loss.requires_grad}")
                print("batch_loss grad_fn:", batch_loss.grad_fn)
                print(f"total_loss.requires_grad = {total_loss.requires_grad}")
                print("batch_loss grad_fn:", total_loss.grad_fn)

            lambda_t.retain_grad()  # Ensure that gradients are retained for lambda_t
            # Backpropagate
            print(f"batch_loss.requires_grad = {batch_loss.requires_grad}")
            print("batch_loss grad_fn:", batch_loss.grad_fn)
            print("before backward lambda_t dtype:", lambda_t.dtype)
            print(f"before backward lambda_t grad_fn: {lambda_t.grad_fn}")

            print("before backward lambda_t gradients:", lambda_t.grad)

            optimizer.zero_grad()
            for p in optimizer.param_groups[0]['params']:
                print(f"{p.shape}: {p.grad}")
            batch_loss.backward(retain_graph=True)  # Retain the graph for further backward passes if needed
            print(f"Batch_loss gradient: {batch_loss.grad}")
            print("After backward lambda_t gradients:", lambda_t.grad)

            print("Lambda_t before step:", lambda_t)
            optimizer.step()
            print("Lambda_t after step:", lambda_t)

            total_train_loss += batch_loss.item()


        # Testing loop (Batch processing for testing)
        for i in range(0, len(test_data), batch_size):
            batch_samples = test_data[i:i + batch_size]
            filenames = [sample[0] for sample in batch_samples]
            neutral_desc_embeddings = [sample[2] for sample in batch_samples]
            stylized_desc_embeddings = [sample[3] for sample in batch_samples]

            # Calculate loss for the batch
            batch_loss = 0
            print("Test: calculate loss")
            for idx, sample in enumerate(batch_samples):
                desc_neutral, desc_stylized = sample[2], sample[3]

                total_loss, lambda_t = compute_loss.clip_loss_alternative(desc_neutral, desc_stylized, lambda_t, T, num_inference_steps, pipe, clip_image_embedder, weight=0.8)

                batch_loss += total_loss.item()

            total_test_loss += batch_loss

        # Print training and testing losses
        avg_train_loss = total_train_loss / len(train_data)
        avg_test_loss = total_test_loss / len(test_data)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        # Store average losses per epoch
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # Save model and losses
        save_dict = {
            "lambda_t": lambda_t.clone().detach().cpu(),  # Move to CPU before saving
            "optimizer_state": optimizer.state_dict(),
            "train_losses": train_losses,
            "test_losses": test_losses
        }
        torch.save(save_dict, save_path)
        print(f"Model saved at: {save_path}")

        # Open in append mode and log data
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([avg_train_loss, avg_test_loss])

        lambda_values = lambda_t.clone().detach().cpu().numpy().tolist()  # Convert tensor to list
        print(lambda_values)
        lambdas.append([lambda_values])
        # Append lambda to CSV
        with open(lambda_log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(lambda_values)  # Write λ values
            print(f"Epoch logged to {log_file} and {lambda_log_file}.")

    return lambda_t, train_losses, test_losses

def load_lambda_model(load_path="./"):
    """
    Loads the saved lambda model.
    """
    if not os.path.exists(load_path):
        print(f"No saved model found at {load_path}!")
        return None, None

    checkpoint = torch.load(load_path, map_location=DEVICE)

    # Ensure lambda is restored as a trainable parameter
    lambda_t = torch.nn.Parameter(checkpoint["lambda_t"].to(DEVICE), requires_grad=True)

    optimizer = torch.optim.Adam([lambda_t])  # Recreate optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(f"Model loaded from: {load_path}")

    return lambda_t, optimizer
