
import torchvision
from torchvision.transforms import v2
import torch
from pycocotools.coco import COCO
from PIL import Image
import os
import matplotlib.pyplot as plt 

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
        image = v2.CenterCrop((256, 256))(image)
        mask = v2.CenterCrop((256, 256))(mask)
        mask = v2.ToDtype(torch.int64)(mask)
        #normalize image
        image = v2.ToDtype(torch.float32, scale=True)(image)
        image = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        return image, mask
    
if __name__ == '__main__':
    coco = coco_dataset()
    img,mask = coco[0]
    #plot image and label side by side
    plt.subplot(1,2,1)
    plt.imshow(img.permute(1,2,0))
    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.show()



