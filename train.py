
import os 

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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


if __name__ == '__main__':
    #load an image segementation model for the coco dataset
    if Baseline:
        model = UNetSmall(im_size, input_channels, num_classes)
    else:
        model = Scale_Model()
    model = model.to(device)
    if pretrained and not Baseline:
        print("Loading pretrained model")
        model.basemodel.load_state_dict(torch.load(load_model_path, weights_only=True))
    # print([name for name,param in model.named_parameters() if "basemodel" not in name])
        optimizer = torch.optim.AdamW([param for name,param in model.named_parameters() if "basemodel" not in name], lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    
 
    #load coco dataset
    #coco = coco_dataset()
    coco = coco_ds_filtered(train=True, filtered_ids=[filtered_ids], scaled=data_is_scaled)
    train_dataloader = torch.utils.data.DataLoader(coco, batch_size=batch_size, shuffle=True, num_workers=4)
    
    #coco_val = coco_dataset(train=False)
    coco_val = coco_ds_filtered(train=False, filtered_ids=[filtered_ids], scaled=data_is_scaled)
    val_dataloader = torch.utils.data.DataLoader(coco_val, batch_size=batch_size, shuffle=False, num_workers=4)
    #train the model
    

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
                images = images.to(device).requires_grad_()
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
                log_dict = {"accuracy": acc.item(),"loss": avg_loss, "lr": scheduler.get_lr()[0]}
                if not Baseline:
                    log_dict["scale_factor"] = model.scale_factor.mean().item()
                    log_dict["scale_factor_loss"] = (model.scale_factor * scale_factor_loss_factor).mean().item()
                    log_dict["rest_loss"] = torch.nn.functional.cross_entropy(outputs, targets).item()
                if step % log_step == 0:
                    plt.subplot(1,4,1)
                    plt.imshow(np.clip(images[0].permute(1,2,0).cpu().detach().numpy()*std + mean,0,1))
                    plt.subplot(1,4,2)
                    plt.imshow(targets[0].cpu().detach().numpy())
                    plt.subplot(1,4,3)
                    plt.imshow(preds[0].long().cpu().detach().numpy())
                    plt.subplot(1,4,4)
                    plt.imshow((preds == targets)[0].float().cpu().detach().numpy())
                    plt.savefig("figures/output"+str(step)+".png")
                    plt.close()
                    log_dict["image"] = wandb.Image("figures/output"+str(step)+".png")
                wandb.log(log_dict)

        
        #validate the model
        model.eval()
        with torch.no_grad():
            mean_acc = 0
            mean_loss = 0
            all_scale_factors = []
            ious = []
            for images, targets in val_dataloader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                _, preds = torch.max(outputs, 1)

                #calculate ious
                iou = []
                for cls in range(num_classes):
                    intersection = ((preds.cpu() == cls) & (targets.cpu() == cls)).float().sum()
                    union = ((preds.cpu() == cls) | (targets.cpu() == cls)).float().sum()
                    iou.append((intersection + 1e-6) / (union + 1e-6))
                ious.append(np.asarray(iou))

                #calculate accuracies
                acc = (preds == targets).float().mean()
                mean_acc += acc.item()
                mean_loss += loss.item()
                if not Baseline:
                    all_scale_factors.append(model.scale_factor)
            mean_acc /= len(val_dataloader)
            mean_loss /= len(val_dataloader)
            log_dict = {"val_accuracy": mean_acc,"val_loss": mean_loss}

            #plot ious
            ious = np.mean(np.asarray(ious), axis=0)
            log_dict[str(filtered_ids)+ " iou"] = ious[filtered_ids]
            log_dict["background iou"] = ious[0]
            plt.bar(range(num_classes), ious)
            plt.ylabel("IoU")
            plt.xlabel("Class ID")
            plt.savefig("figures/ious_eval"+str(epoch)+".png")
            plt.close()
            log_dict["ious"] = wandb.Image("figures/ious_eval"+str(epoch)+".png")

            if not Baseline:
                #create a histogramm of all scales
                all_scale_factors = torch.cat(all_scale_factors)
                plt.hist(all_scale_factors.cpu().detach().numpy())
                plt.savefig("figures/scale_hist"+str(epoch)+".png")
                plt.close()
                log_dict["scale_hist"] = wandb.Image("figures/scale_hist"+str(epoch)+".png")
            wandb.log(log_dict)
            print(f"Epoch: {epoch}, Val Accuracy: {mean_acc}, Val Loss: {mean_loss}")
            if mean_loss < min_eval_loss and Baseline:
                print("Saving model")
                torch.save(model.state_dict(), save_model_path)
                min_eval_loss = mean_loss