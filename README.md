# GNNS final project: Disentanglement in medical images with diffusion models
Final Project for lecture Generative Neural Networks for the Sciences


## Organization
### data: 
  - contains the .csv-file with labels for the full PADCHEST dataset
  - a small sample of our custom .pkl-file with 10 instances
  - a script to create the .pkl-file from the downloaded PADCHEST data

### lambda_training: first approach to optimize lambda for disentanglement
  - main_colab.ipynb contains all the code that is distributed among the subfolders and .py-files and is also the file we used to conduct our experiments
  - main.py contains functions to run training, re-training, inference and data generation
    
  - embedding/embedding.py: process the .pkl-file, generate text descriptions from labels and then create text and image embeddings based on the data
    
  - training/compute_loss.py: contains all loss functions used
  - training/gen_img_from_pretrained.py: contains functions for image generation with the stable diffusion pipeline and the custom dataset, as well as training functions
  - training/lambda_model.pth: the trained model after 10 epochs with T=6 steps

  - logs: contains train/test loss log after 10 epochs and lambda log for 10 epochs and T=6 steps

  - plots: contains the plots based on the logs which are also used in the report as well as the .py-file with the plotting functions

  - generated_imgs: contains all visualizations of different lambda schedules, as well as the comparison before and after training with T=6 which was used in the report
