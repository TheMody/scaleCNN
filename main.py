

import torch
import numpy as np
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm
from dataset import coco_dataset
from unets import UNet
import wandb
from utils import CosineWarmupScheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    #load an image segementation model for the coco dataset
    max_epochs = 10
    lr = 5e-4
    model = UNet(256, 3, 91)
    model = model.to(device)
    #load coco dataset
    coco = coco_dataset()
    train_dataloader = torch.utils.data.DataLoader(coco, batch_size=8, shuffle=True, num_workers=4)
    
    coco_val = coco_dataset(train=False)
    val_dataloader = torch.utils.data.DataLoader(coco_val, batch_size=8, shuffle=False, num_workers=4)
    #train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=max_epochs*len(train_dataloader))
    wandb.init(project="image_segmentation_scalefree")

    step = 0
    for epoch in range(max_epochs):
        model.train()
        for images, targets in (pbar := tqdm(train_dataloader)):
            step += 1
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=loss.item())
            _, preds = torch.max(outputs, 1)
            acc = (preds == targets).float().mean()
            wandb.log({"accuracy": acc.item(),"loss": loss.item(), "lr": scheduler.get_lr()[0]})
         #   if step % 100 == 0:
                #compute iou score for all classes

        
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