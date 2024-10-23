

import torch
from config import *
from unets import UNetSmall

import torch.nn.functional as F

def differentiable_interpolate(x, scale_factors):
    """
    Interpolates the input tensor x by scale_factors, allowing backpropagation through scale_factors.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W).
        scale_factors (torch.Tensor): Scaling factors for height and width, shape (2,), requires_grad=True.
        out_size (tuple, optional): Desired output size (H_out, W_out). If None, uses input size.

    Returns:
        torch.Tensor: Interpolated tensor.
    """
    N, C, H, W = x.shape
    device = x.device
    dtype = x.dtype
    #without any gradients create a grid of size H_out, W_out
    with torch.no_grad():
        H_out, W_out = int(H * scale_factors[0]), int(W * scale_factors[1])

            # Create normalized grid coordinates ranging from -1 to 1
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-scale_factors[0], scale_factors[0], H_out, device=device, dtype=dtype),
            torch.linspace(-scale_factors[1], scale_factors[1], W_out, device=device, dtype=dtype)
        )
        grid = torch.stack((grid_x, grid_y), dim=2)  # Shape: (H_out, W_out, 2)


    # Adjust grid coordinates inversely proportional to scale_factors
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  # Shape: (N, H_out, W_out, 2)
    inv_scale = 1 / scale_factors.view(-1, 1, 1, 2)  # Shape: (N, 1, 1, 2)
    grid = grid * inv_scale  # Scale grid to simulate interpolation

    # Perform grid sampling i.e. fill the output tensor with values from the input tensor
    y = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return y

def image_to_patches(image, patch_size):
    # Reshape image: (batch_size, channels, height, width) -> (batch_size, patches, patch_size * patch_size * channels)
    batch_size, channels, height, width = image.shape
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, patch_size * patch_size * channels)
    return patches

class Scale_Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.basemodel = UNetSmall(im_size, input_channels, num_classes)
        if scale_model == "transformer":
            self.patch_size = 16
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.patch_embeddings = torch.nn.Linear(input_channels * self.patch_size ** 2, hidden_dim)
            self.scale_model = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2,dim_feedforward=4*hidden_dim, batch_first=True, activation='gelu') 
        if scale_model == "cnn":
            self.scale_model = torch.nn.Sequential(torch.nn.Conv2d(input_channels, hidden_dim, kernel_size), torch.nn.ReLU(),torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size),torch.nn.ReLU())
        self.linear_scale = torch.nn.Linear(hidden_dim, 1)

    #x is (batchsize, channels, height, width)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(f"Before scale: {x.requires_grad}")
        init_shapes = x.shape
        input = x
        if scale_model == "cnn":
            x = self.scale_model(x)
            x = torch.sum(x, dim=(2, 3)) / (x.shape[2] * x.shape[3])
        if scale_model == "transformer":
            # x is (batchsize, channels, height, width)
            x = image_to_patches(x, self.patch_size)
            x = self.patch_embeddings(x)
            #add the cls token
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = self.scale_model(x)[:,0]

        x = torch.nn.functional.sigmoid(self.linear_scale(x))*3 + 16/im_size
        #print(f"Before interpolate: {x.requires_grad}")
        # x is (batchsize, scale) = (1,1)
        input = differentiable_interpolate(input, torch.cat((x,x), dim = 1)[0])
        self.scale_factor = x[0]


        #pad input to length divisble by 2^4 needed fpr unet
        pad = 2**4 - (input.shape[2] % 2**4)
        input = torch.nn.functional.pad(input, (0, pad, 0, pad))
        

        #print(f"Before unet: {x.requires_grad}")
        x = self.basemodel(input)
        #print(f"After unet: {x.requires_grad}")
        x= torch.nn.functional.interpolate(x, size=(init_shapes[2],init_shapes[3]), mode = "bilinear")
        #print(f"After scaling back: {x.requires_grad}")
        return x
    
if __name__ == '__main__':  
    import matplotlib.pyplot as plt

    # Load the image
    image = plt.imread('testimg.jpg')

    # Convert the image to a torch tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float()
    scale = 0.5
    # Define the scale factors
    scale = torch.tensor([scale, scale], requires_grad=True)

    # Interpolate the image
    interpolated_image = differentiable_interpolate(image_tensor, scale_factors=scale)

    # Convert the interpolated image tensor to a numpy array
    interpolated_image_np = (interpolated_image/255.0).squeeze(0).permute(1, 2, 0).detach().numpy()

    # Visualize the original and interpolated images
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[1].imshow(interpolated_image_np)
    axes[1].set_title('Interpolated Image')
    plt.show()


    #scale = torch.tensor((scale,scale), requires_grad=True)
    x = torch.randn(1,3,256,256, requires_grad=True)
    # scaler = torch.nn.Upsample(scale_factor=scale[0])
    # y = scaler(x)
    y = differentiable_interpolate(x,scale_factors=scale)
    L  = y.sum()
    L.backward()
    #y.backward()
    print(scale.grad)