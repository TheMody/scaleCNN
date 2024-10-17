

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

class Scale_Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = UNetSmall(im_size, input_channels, num_classes)
        self.scale_model = torch.nn.Sequential(torch.nn.Conv2d(input_channels, 64, 5), torch.nn.ReLU(),torch.nn.Conv2d(64, 64, 5),torch.nn.ReLU())
        self.linear_scale = torch.nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.scale_model(x)
        x = torch.sum(x, dim=(2, 3)) / (x.shape[2] * x.shape[3])
        x = torch.nn.functional.sigmoid(self.linear_scale(x))*3 + 16/im_size
        #x = torch.clip(x, 16/im_size, 3)
        # x is (batchsize, scale) = (1,1)
        input = differentiable_interpolate(input, torch.cat((x,x), dim = 1)[0])
        self.scale_factor = x[0]
        #input= torch.nn.functional.interpolate(input, size=(int(im_size*x[0]),int(im_size*x[0])))
        #input = torch.nn.Upsample(scale_factor=x)(input)
        #pad input to length divisble by 2^4 
        pad = 2**4 - (input.shape[2] % 2**4)
        input = torch.nn.functional.pad(input, (0, pad, 0, pad))
        x = self.model(input)
        x= torch.nn.functional.interpolate(x, size=(im_size,im_size), mode = "bilinear")
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
    y = differentiable_interpolate(x,scale_factors=scale)
    L  = y.sum()
    L.backward()
    #y.backward()
    print(scale.grad)