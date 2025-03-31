# File: cVAE-Dict/train.py
import torch
import torch.nn.functional as F

def onehot(labels, num_classes):
    return F.one_hot(labels, num_classes).float()

def train_epoch(model, dataloader, optimizer, lambda_sparse, device):
    model.train()
    total_loss = 0
    # for batch in dataloader:
    #     print("cond type:", type(batch["cond"]))
    #     print("cond sample:", batch["cond"])
    #     break
    for batch in dataloader:
        x = batch["image"].to(device)
        # c_src = onehot(torch.tensor(batch["cond"], device=device), num_classes=2)
        c_src = onehot(batch["cond"].to(device), num_classes=2)
        c_tgt = c_src  # for reconstruction

        x_hat, alpha = model(x, c_src, c_tgt)
        loss_recon = torch.nn.functional.mse_loss(x_hat, x)
        loss_sparse = torch.norm(alpha, p=1)
        loss = loss_recon + lambda_sparse * loss_sparse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

