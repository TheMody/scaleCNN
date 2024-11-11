
import torch
import numpy as np
from tqdm import tqdm
from dataset import coco_dataset,coco_ds_filtered
from unets import UNet, UNetSmall
import wandb
from utils import CosineWarmupScheduler
from model import Scale_Model
import matplotlib.pyplot as plt
from config import *


def eval(Baseline):
    coco_val = coco_ds_filtered(train=False, filtered_ids=[filtered_ids], scaled=data_is_scaled)
    val_dataloader = torch.utils.data.DataLoader(coco_val, batch_size=batch_size, shuffle=False, num_workers=4)

    if Baseline:
        model = UNetSmall(im_size, input_channels, num_classes)
        model.load_state_dict(torch.load(load_model_path, weights_only=True))
    else:
        model = Scale_Model()
        model.basemodel.load_state_dict(torch.load(load_model_path, weights_only=True))
        
    wandb.init(project="image_segmentation_scalefree", config=config)

    model = model.to(device)
    #validate the model
    model.eval()
    with torch.no_grad():
        mean_acc = 0
        mean_loss = 0
        all_scale_factors = []
        ious = []
        for images, targets in tqdm(val_dataloader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            _, preds = torch.max(outputs, 1)
            acc = (preds == targets).float().mean()
            
            iou = []
            for cls in range(num_classes):
                intersection = ((preds.cpu() == cls) & (targets.cpu() == cls)).float().sum()
                union = ((preds.cpu() == cls) | (targets.cpu() == cls)).float().sum()
                iou.append((intersection + 1e-6) / (union + 1e-6))
            ious.append(np.asarray(iou))
            mean_acc += acc.item()
            mean_loss += loss.item()
            if not Baseline:
                all_scale_factors.append(model.scale_factor)

        ious = np.mean(np.asarray(ious), axis=0)
        print("background iou:",ious[0])  
        print(str(filtered_ids)+" iou:",ious[filtered_ids])
        wandb.log({str(filtered_ids)+" iou without scaling":ious[filtered_ids]})
        #do a barplot of all ious
        plt.bar(range(num_classes), ious)
        plt.ylabel("IoU")
        plt.xlabel("Class ID")
        plt.savefig("figures/ious_eval.png")

        mean_acc /= len(val_dataloader)
        mean_loss /= len(val_dataloader)
        if not Baseline:
            #create a histogramm of all scales
            all_scale_factors = torch.cat(all_scale_factors)
            plt.hist(all_scale_factors.cpu().detach().numpy())
            plt.savefig("figures/scale_hist_eval.png")
        print(f" Val Accuracy: {mean_acc}, Val Loss: {mean_loss}")
