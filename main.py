

import torch
import numpy as np
from tqdm import tqdm
from dataset import coco_dataset
from unets import UNet, UNetSmall
import wandb
from utils import CosineWarmupScheduler
from model import Scale_Model
import matplotlib.pyplot as plt
from config import *


if __name__ == '__main__':
    #load an image segementation model for the coco dataset
    if Baseline:
        model = UNetSmall(im_size, input_channels, num_classes)
    else:
        model = Scale_Model()
    model = model.to(device)
    #load coco dataset
    coco = coco_dataset()
    train_dataloader = torch.utils.data.DataLoader(coco, batch_size=batch_size, shuffle=True, num_workers=4)
    
    coco_val = coco_dataset(train=False)
    val_dataloader = torch.utils.data.DataLoader(coco_val, batch_size=batch_size, shuffle=False, num_workers=4)
    #train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=float(max_epochs*(len(train_dataloader)//microbatch_size)))
    wandb.init(project="image_segmentation_scalefree", config=config)
    
    min_eval_loss = 1e10
    step = 0
    for epoch in range(max_epochs):
        model.train()
        trainset = iter(train_dataloader)
        for i in (pbar := tqdm(range(len(train_dataloader)//microbatch_size))):
            step += 1
            optimizer.zero_grad()
            avg_loss = 0
            for _ in range(microbatch_size):
                images, targets = next(trainset)
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs, targets) 
                if not Baseline:
                    loss += (model.scale_factor * scale_factor_loss_factor).mean()
                loss = loss / microbatch_size
                loss.backward()
                avg_loss += loss.item()

            #add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                pbar.set_postfix(loss=avg_loss)
                _, preds = torch.max(outputs, 1)
                acc = (preds == targets).float().mean()
                if step % log_step == 0:
                    plt.subplot(1,3,1)
                    plt.imshow(np.clip(images[0].permute(1,2,0).cpu().detach().numpy()*std + mean,0,1))
                    plt.subplot(1,3,2)
                    plt.imshow(targets[0].cpu().detach().numpy())
                    plt.subplot(1,3,3)
                    plt.imshow(preds[0].cpu().detach().numpy())
                    plt.savefig("figures/output"+str(step)+".png")
                    wandb.log({"accuracy": acc.item(),"loss": avg_loss, "lr": scheduler.get_lr()[0],"image":wandb.Image("figures/output"+str(step)+".png"), "scale_factor": model.scale_factor.item()})
                else:
                    wandb.log({"accuracy": acc.item(),"loss": avg_loss, "lr": scheduler.get_lr()[0], "scale_factor": model.scale_factor.item()})

        
        #validate the model
        model.eval()
        with torch.no_grad():
            mean_acc = 0
            mean_loss = 0
            for images, targets in val_dataloader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                _, preds = torch.max(outputs, 1)
                acc = (preds == targets).float().mean()
                mean_acc += acc.item()
                mean_loss += loss.item()
            mean_acc /= len(val_dataloader)
            mean_loss /= len(val_dataloader)
            wandb.log({"val_accuracy": mean_acc,"val_loss": mean_loss})
            print(f"Epoch: {epoch}, Val Accuracy: {mean_acc}, Val Loss: {mean_loss}")
            if mean_loss < min_eval_loss:
                torch.save(model.state_dict(), "model.pth")
                min_eval_loss = mean_loss