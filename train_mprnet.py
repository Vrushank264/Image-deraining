import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import MPRNet
from dataset import RainDataset
from losses import CharbonnierLoss, EdgeLoss


def train(loader, model, c_loss, edge_loss, mse_loss, opt, scheduler, epoch, writer, step, device = torch.device('cuda')):
    
    train_loss, train_psnr = [], []
    epoch_loss, epoch_psnr, total_psnr = 0.0, 0.0, 0.0
    loop = tqdm(loader, position = 0, leave = True)
    model.train()

    for idx, (ip, gt) in enumerate(loop):
        
        ip = ip.to(device)
        gt = gt.to(device)
        
        opt.zero_grad()
        
        pred = model(ip)
        
        loss_c = np.sum([c_loss(pred[j], gt) for j in range(len(pred))])
        loss_e = np.sum([edge_loss(pred[j], gt) for j in range(len(pred))])
        loss = (loss_c) + (0.05 * loss_e)
    
        loss.backward()
        opt.step()
        
        train_loss.append(loss.item())
        pred1, pred2, pred3 = pred
        total_psnr = 10 * torch.log10(1.0 / mse_loss(pred1, gt))
        train_psnr.append(total_psnr)
        
        epoch_loss = sum(train_loss)/len(train_loss)
        epoch_psnr = sum(train_psnr)/len(train_psnr)
        
        model.eval()
        
        writer.add_scalar("Training/loss", epoch_loss, global_step = step)
        writer.add_scalar("Training/PSNR", epoch_psnr, global_step = step)
        
        if idx % 100 == 0:
            with torch.no_grad():
                gen_img3, gen_img2, gen_img1 = model(fixed_ip)
                writer.add_image("Observation/generated_image_at_stage1", gen_img1.squeeze(0), global_step = step)
                writer.add_image("Observation/generated_image_at_stage2", gen_img2.squeeze(0), global_step = step)
                writer.add_image("Observation/generated_image_at_stage3", gen_img3.squeeze(0), global_step = step)
        
        step+=1
        torch.cuda.empty_cache()
        
    scheduler.step()
    print(f'\nEpoch: {epoch}, Loss: {epoch_loss} and PSNR: {epoch_psnr}\n')
    return step


def validate(loader, model, c_loss, edge_loss, mse_loss, epoch, writer, val_step, device = torch.device('cuda')):
    
    val_loss, val_psnr = [], []
    loss, psnr = 0.0, 0.0
    loop = tqdm(loader, position = 0, leave = True)
    print("Validating...")
    model.eval()
    
    with torch.no_grad():
      for i, (ip, gt) in enumerate(loop):
          
          ip = ip.to(device)
          gt = gt.to(device)
          
          pred = model(ip)
          loss_c = np.sum([c_loss(pred[j], gt) for j in range(len(pred))])
          loss_e = np.sum([edge_loss(pred[j], gt) for j in range(len(pred))])
          loss = (loss_c) + (0.05 * loss_e)
          
          val_loss.append(loss.item())
          pred1, pred2, pred3 = pred
          psnr = 10 * torch.log10(1.0 / mse_loss(pred1, gt))
          val_psnr.append(psnr)
          
          val_step += 1
          
    loss = sum(val_loss)/len(val_loss)
    psnr = sum(val_psnr)/len(val_psnr)
      
    writer.add_scalar("Validation/loss", loss, global_step = val_step)
    writer.add_scalar("Validation/PSNR", psnr, global_step = val_step)
      
      
    gen_img3, gen_img2, gen_img1 = model(fixed_ip1)
    writer.add_image("Valid_image/generated_image_at_stage1", gen_img1.squeeze(0), global_step = val_step)
    writer.add_image("Valid_image/generated_image_at_stage2", gen_img2.squeeze(0), global_step = val_step)
    writer.add_image("Valid_image/generated_image_at_stage3", gen_img3.squeeze(0), global_step = val_step)
    
    print(f'Validation Loss: {loss} and Validation PSNR: {psnr}\n')
    return val_step


def main():
    
    TRAIN_DATA_IP = '/content/Dataset/input'
    TRAIN_DATA_GT = '/content/Dataset/target'
    VALID_DATA_IP = '/content/Dataset/Test/input'
    VALID_DATA_GT = '/content/Dataset/Test/target'
    MODEL_DIR = '/content/drive/MyDrive/derain'
    LOGS_DIR = '/content/drive/MyDrive/derain/logs'
    
    device = torch.device('cuda')
    model = MPRNet().to(device)
    train_data = RainDataset(gt_path = TRAIN_DATA_GT, ip_path = TRAIN_DATA_IP)
    valid_data = RainDataset(gt_path = VALID_DATA_GT, ip_path = VALID_DATA_IP, img_size = 128)
    train_loader = DataLoader(train_data, batch_size = 16, shuffle = True, num_workers = 2)
    valid_loader = DataLoader(valid_data, batch_size = 10, shuffle = False, num_workers = 2)
    
    c_loss = CharbonnierLoss()
    edge_loss = EdgeLoss()
    mse_loss = nn.MSELoss()

    opt = torch.optim.Adam(model.parameters(), lr = 1e-5, betas = (0.9, 0.999), weight_decay = 1e-8, eps = 1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 1e-6)
    
    global fixed_ip
    fixed_ip = T.Compose([T.Resize((256, 256)), T.ToTensor()])(Image.open('/content/Dataset/input/10049.jpg')).unsqueeze(0)
    fixed_ip = fixed_ip.to(device)

    global fixed_ip1
    fixed_ip1 = T.Compose([T.Resize((256,256)), T.ToTensor()])(Image.open('/content/Dataset/Test/input/26.png')).unsqueeze(0)
    fixed_ip1 = fixed_ip1.to(device)

    target = T.Compose([T.Resize((256, 256)), T.ToTensor()])(Image.open('/content/Dataset/target/10049.jpg'))
    target = target.to(device)
    target1 = T.Compose([T.Resize((256, 256)), T.ToTensor()])(Image.open('/content/Dataset/Test/target/26.png'))
    target1 = target1.to(device)
    
    writer = SummaryWriter(LOGS_DIR)
    step, val_step = 0, 0
    
    
    for epoch in range(100):

        print(scheduler.get_last_lr())
        val_step = validate(valid_loader, model, c_loss, edge_loss, mse_loss, epoch, writer, val_step)
        step = train(train_loader, model, c_loss, edge_loss, mse_loss, opt, scheduler, epoch, writer, step)
        scheduler.step()
        torch.save(model.state_dict(), open(MODEL_DIR + 'model.pth', 'wb'))
        
             
if __name__ == '__main__':

    main()    
    