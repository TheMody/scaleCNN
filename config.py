
import torch

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
lr=0.001
max_epochs=10
batch_size=1
microbatch_size = 32
num_classes = 91
im_size = 256
log_step = 100
input_channels = 3
hidden_dim = 64
kernel_size = 5
scale_factor_loss_factor = 1e-1
Baseline = False
scale_model =  "transformer"#"cnn"#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {"mean":mean, "std":std, "lr":lr, "max_epochs":max_epochs, "batch_size":batch_size, "num_classes":num_classes, "im_size":im_size, "log_step":log_step, "input_channels":input_channels, "device":device, "hidden_dim":hidden_dim, "kernel_size":kernel_size, "scale_factor_loss_factor":scale_factor_loss_factor, "Baseline":Baseline, "scale_model":scale_model}
