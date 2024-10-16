
import torch

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
lr=0.001
max_epochs=10
batch_size=8
num_classes = 91
im_size = 256
log_step = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')