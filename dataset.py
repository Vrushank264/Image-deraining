import torchvision.transforms as T
from torch.utils.data import Dataset
import os
from PIL import Image


class RainDataset(Dataset):
    
    def __init__(self, gt_path, ip_path, img_size = 64):
        
        super().__init__()
        
        self.gt_fnames = [os.path.join(gt_path, f) for f in sorted(os.listdir(gt_path))]
        self.ip_fnames = [os.path.join(ip_path, f) for f in sorted(os.listdir(ip_path))]
        
        self.transform = T.Compose([T.CenterCrop((img_size, img_size)),
                                    T.ToTensor(),
                                    T.Normalize([0.0,0.0,0.0],[1.0,1.0,1.0])])
        
    def __len__(self):
        
        return len(self.gt_fnames)
    
    def __getitem__(self, idx):
        
        ip_img = Image.open(self.ip_fnames[idx]).convert('RGB')
        gt_img = Image.open(self.gt_fnames[idx]).convert('RGB')
        ip_img = self.transform(ip_img)
        gt_img = self.transform(gt_img)
        
        return ip_img, gt_img
    
    
class TestDataset(Dataset):
    
    def __init__(self, ip_path, img_size = 64):
        
        super().__init__()
        
        self.ip_fnames = [os.path.join(ip_path, f) for f in sorted(os.listdir(ip_path))]
        
        self.transform = T.Compose([T.CenterCrop((img_size, img_size)),
                                    T.ToTensor(),
                                    T.Normalize([0.0,0.0,0.0],[1.0,1.0,1.0])])
        
    def __len__(self):
        
        return len(self.gt_fnames)
    
    def __getitem__(self, idx):
        
        ip_img = Image.open(self.ip_fnames[idx]).convert('RGB')
        ip_img = self.transform(ip_img)
        
        return ip_img
        
