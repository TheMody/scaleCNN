
from torchvision.transforms import v2
import torch
from pycocotools.coco import COCO
from PIL import Image
import os
import matplotlib.pyplot as plt 
from config import *


class coco_ds_filtered(torch.utils.data.Dataset):
    def __init__(self,train,filtered_ids):
        if train:
            self.coco = COCO('data/train/instances_train2017.json')
            self.img_dir = 'data/train/train2017/'
        else:
            self.coco = COCO('data/val/instances_val2017.json')
            self.img_dir = 'data/val/val2017/'
        
        self.filtered_ids = filtered_ids
        self.cat_ids = self.coco.getCatIds()
        self.idx_to_id =[int(s.split(".jpg")[0]) for s in  os.listdir(self.img_dir)]
        self.length = len(os.listdir(self.img_dir))

        print("filtering imgs")
        cat_ids = self.coco.getImgIds(imgIds=self.idx_to_id,catIds=filtered_ids)
        #get the indices of the filtered images
        self.idx_to_id = [self.idx_to_id[i] for i in range(len(self.idx_to_id)) if self.idx_to_id[i] in cat_ids]
        print("finished filtering imgs")

    def __len__(self):
        return self.length 
    
    def __getitem__(self, idx):
        img = self.coco.imgs[self.idx_to_id[idx]]
        image = Image.open(os.path.join(self.img_dir, img['file_name']))
        anns_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)
        
        mask = torch.zeros((image.size[1], image.size[0]))
        # for i in range(0,len(anns)):
        #     mask = torch.max(mask,torch.Tensor(self.coco.annToMask(anns[i])*anns[i]['category_id']))

        #get the bounding box of the filtered category
        for id in self.filtered_ids:
            anns_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=id, iscrowd=None)
            anns = self.coco.loadAnns(anns_ids)
            for i in range(0,len(anns)):
                mask = torch.max(mask,torch.Tensor(self.coco.annToMask(anns[i])*anns[i]['category_id']))

        bbox = anns[i]['bbox']

        mask = torch.Tensor(mask)
        image = v2.PILToTensor()(image)

        image = image[:,int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]
        mask = mask[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]

        #resize image largest dim to im_size
        if image.shape[1] > image.shape[2]:
            mask = v2.Resize((im_size,im_size*image.shape[2]//image.shape[1]),interpolation=v2.InterpolationMode.NEAREST)(mask.unsqueeze(0)).squeeze(0)
            image = v2.Resize((im_size,im_size*image.shape[2]//image.shape[1]))(image)
            
            #pad to im_size,im_size
            pad = im_size - image.shape[2]
            image = torch.nn.functional.pad(image, (0, pad, 0, 0))
            mask = torch.nn.functional.pad(mask, (0, pad, 0, 0))
        else:
            mask = v2.Resize((im_size*image.shape[1]//image.shape[2],im_size),interpolation=v2.InterpolationMode.NEAREST)(mask.unsqueeze(0)).squeeze(0)
            image = v2.Resize((im_size*image.shape[1]//image.shape[2],im_size))(image)
            
            #pad to im_size,im_size
            pad = im_size - image.shape[1]
            image = torch.nn.functional.pad(image, (0, 0, 0, pad))
            mask = torch.nn.functional.pad(mask, (0, 0, 0, pad))

        
        #center crop both images equally
       # image = v2.CenterCrop((im_size, im_size))(image)
       # mask = v2.CenterCrop((im_size, im_size))(mask)
        mask = v2.ToDtype(torch.int64)(mask)
        #normalize image
        image = v2.ToDtype(torch.float32, scale=True)(image)
        image = v2.Normalize(mean=mean, std=std)(image)
        return image, mask

class coco_dataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        if train:
            self.coco = COCO('data/train/instances_train2017.json')
            self.img_dir = 'data/train/train2017/'
        else:
            self.coco = COCO('data/val/instances_val2017.json')
            self.img_dir = 'data/val/val2017/'
            
        self.cat_ids = self.coco.getCatIds()
        self.idx_to_id =[int(s.split(".jpg")[0]) for s in  os.listdir(self.img_dir)]
        self.length = len(os.listdir(self.img_dir))
  

    def __len__(self):
        return self.length 
    
    def __getitem__(self, idx):
        img = self.coco.imgs[self.idx_to_id[idx]]
        image = Image.open(os.path.join(self.img_dir, img['file_name']))
        anns_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)
        
        mask = torch.zeros((image.size[1], image.size[0]))
        for i in range(0,len(anns)):
            mask = torch.max(mask,torch.Tensor(self.coco.annToMask(anns[i])*anns[i]['category_id']))

        mask = torch.Tensor(mask)
        image = v2.PILToTensor()(image)

        #center crop both images equally
        image = v2.CenterCrop((im_size, im_size))(image)
        mask = v2.CenterCrop((im_size, im_size))(mask)
        mask = v2.ToDtype(torch.int64)(mask)
        #normalize image
        image = v2.ToDtype(torch.float32, scale=True)(image)
        image = v2.Normalize(mean=mean, std=std)(image)
        return image, mask
    
if __name__ == '__main__':
    coco = coco_ds_filtered(True,[35])
    for i in range(100):
        img,mask = coco[i]
        #plot image and label side by side
        plt.subplot(1,2,1)
        plt.imshow(img.permute(1,2,0))
        plt.subplot(1,2,2)
        plt.imshow(mask)
        plt.show()



