import torch
import torch.nn as nn
import torch.nn.functional as fun


class CharbonnierLoss(nn.Module):
    
    def __init__(self, eps = 1e-3):
        
        super().__init__()
        self.eps = eps
        
    def forward(self, x, y):
        
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss
    
    
class EdgeLoss(nn.Module):
    
    def __init__(self, device = torch.device('cuda')):
        
        super().__init__()
        k = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3,1,1,1)
        self.kernel = self.kernel.to(device)
        self.c_loss = CharbonnierLoss()
        
    def gaussian_conv(self, img):
        
        num_c, _, kw, kh = self.kernel.shape
        img = fun.pad(img, (kw//2, kh//2, kw//2, kh//2), mode = 'replicate')
        return fun.conv2d(img, self.kernel, groups = num_c)
    
    def laplacian_kernel(self, x):
        
        x1 = self.gaussian_conv(x)
        downsample = x1[:, :, ::2, ::2]
        new_filter = torch.zeros_like(x1)
        new_filter[:, :, ::2, ::2] = downsample*4
        x1 = self.gaussian_conv(new_filter)
        diff = x - x1
        return diff
    
    def forward(self, x, y):
        
        loss = self.c_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
    
    
