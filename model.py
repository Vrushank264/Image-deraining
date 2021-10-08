import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    
    '''

    Channel Attention(Squeeze and Excitation Operation): 

                    HxWxC
         -------------|       
         '           GAP     
         '            |  
         '          1x1xC
         '            |
         '        Conv + ReLU 
         '            |
         '          1x1xC/r (r = reduction ratio)
         '            |
         '          Conv
         '            |
         '          1x1xC
         '            |
         '         Sigmoid
         '            |
         -------------*
                      |
                     out
                                                                                  
    Multiplying 1x1xC with input again gives output -> HxWxC

    '''
    
    def __init__(self, channels, r = 16, bias = False):
        
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(nn.Conv2d(channels, channels//r, kernel_size = 1, padding = 0, bias = bias),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(channels//r, channels, kernel_size = 1, padding = 0, bias = bias),
                                    nn.Sigmoid())
        
    def forward(self, x):
        
        out = self.squeeze(x)
        out = self.excite(out)
        return x * out
    

class CAB(nn.Module):
    
    '''
    
    Channel Attention Block:
        
            HxWxC
              |---------------
       Conv + PReLU + Conv   '
              |              '
        ChannelAttention()   '
              |              '
              +--------------'
              |
             out
       
    '''
    
    def __init__(self, in_c, r, bias):
        
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size = 3, padding = 1, bias = bias),
                                  nn.PReLU(),
                                  nn.Conv2d(in_c, in_c, kernel_size = 3, padding = 1, bias = bias))
        
        self.ca = ChannelAttention(in_c, r = r, bias = bias)
        
    def forward(self, x):
        
        out = self.body(x)
        out = self.ca(out)
        out += x
        return out
    
    
class SAM(nn.Module):
    
    '''
    
    Supervised Attention Module(SAM):
        
                       -------------------------------------------
                      '                                           '
        HxWxC ------------------> Conv1 ------------------> * --> + ----> HxWxC (Output) 
   (Features from     '                                     '
    previous layers)  '                                     '
                     Conv2                                  '
                      '                                     '
        HxWx3 ------- + -> HxWx3 --> Conv3 --> Sigmoid --> HxWxC 
       (Input)           (Restored                    (Attention Maps)
                          Image)
        
      
    '''
    
    def __init__(self, in_c, bias):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size = 3, padding = 1, bias = bias)
        self.conv2 = nn.Conv2d(in_c, 3, kernel_size = 3, padding = 1, bias = bias)
        self.conv3 = nn.Conv2d(3, in_c, kernel_size = 3, padding = 1, bias = bias)
        
    def forward(self, x, ip_img):
        
        x1 = self.conv1(x)
        restored_img = self.conv2(x) + ip_img
        attn_maps = torch.sigmoid(self.conv3(restored_img))
        x1 = x1 * attn_maps
        x1 += x
        
        return x1, restored_img
    
    
class UpSample(nn.Module):
    
    def __init__(self, in_c, factor):
        
        super().__init__()
        self.up_block = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
                                   nn.Conv2d(in_c + factor, in_c, kernel_size = 1, bias = False))
        
    def forward(self, x):
        
        return self.up_block(x)
    

class DownSample(nn.Module):
    
    def __init__(self, in_c, factor):
        
        super().__init__()
        self.down_block = nn.Sequential(nn.Upsample(scale_factor = 0.5, mode = 'bilinear', align_corners = False),
                                        nn.Conv2d(in_c, in_c + factor, kernel_size = 1, bias = False))
        
    def forward(self, x):
        
        return self.down_block(x)
    
    
class SkipUpSample(nn.Module):
    
    def __init__(self, in_c, factor):
        
        super().__init__()
        self.skip_block = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
                                   nn.Conv2d(in_c + factor, in_c, kernel_size = 1, bias = False))
        
    def forward(self, x, y):
        
        x = self.skip_block(x)
        x += y
        return x
    

class Encoder(nn.Module):
    
    '''
    UNet like Encoder design with CAB(Channel Attention Block)
    as building blocks and skip connections.
    
    '''
        
    def __init__(self, in_c, out_c, r, csff, bias):
        
        super().__init__()
        
        self.enc_block1 = nn.Sequential(CAB(in_c = in_c , r = r, bias = bias),
                                        CAB(in_c = in_c , r = r, bias = bias)) 
        self.enc_block2 = nn.Sequential(CAB(in_c + out_c, r, bias),
                                        CAB(in_c + out_c, r, bias))
        self.enc_block3 = nn.Sequential(CAB(in_c + out_c + out_c, r, bias),
                                        CAB(in_c + out_c + out_c, r, bias))
        
        self.down_1_to_2 = DownSample(in_c, out_c)
        self.down_2_to_3 = DownSample(in_c + out_c, out_c)
        
        if csff:
            
            self.enc1_csff = nn.Conv2d(in_c, in_c, kernel_size = 1, bias = bias)
            self.enc2_csff = nn.Conv2d(in_c + out_c, in_c + out_c, kernel_size = 1, bias = bias)
            self.enc3_csff = nn.Conv2d(in_c + out_c + out_c, in_c + out_c + out_c, kernel_size = 1, bias = bias)
            
            self.dec1_csff = nn.Conv2d(in_c, in_c, kernel_size = 1, bias = bias)
            self.dec2_csff = nn.Conv2d(in_c + out_c, in_c + out_c, kernel_size = 1, bias = bias)
            self.dec3_csff = nn.Conv2d(in_c + out_c + out_c, in_c + out_c + out_c, kernel_size = 1, bias = bias)
        
        
    def forward(self, x, encoder_out = None, decoder_out = None):
        
        condition = (encoder_out is not None) and (decoder_out is not None)
        out1 = self.enc_block1(x)
        
        if condition is True:
            out1 = out1 + self.enc1_csff(encoder_out[0]) + self.dec1_csff(decoder_out[0])
            
        x = self.down_1_to_2(out1)
        out2 = self.enc_block2(x)
        
        if condition is True:
            out2 = out2 + self.enc2_csff(encoder_out[1]) + self.dec2_csff(decoder_out[1])   
            
        x = self.down_2_to_3(out2)
        out3 = self.enc_block3(x)
        
        if condition is True:
            out3 = out3 + self.enc3_csff(encoder_out[2]) + self.dec3_csff(decoder_out[2])
            
        return [out1, out2, out3]
    
    
class Decoder(nn.Module):
    
    def __init__(self, in_c, out_c, r, bias):
        
        super().__init__()
        self.dec_block1 = nn.Sequential(CAB(in_c = in_c , r = r, bias = bias),
                                        CAB(in_c = in_c , r = r, bias = bias)) 
        self.dec_block2 = nn.Sequential(CAB(in_c + out_c, r, bias),
                                        CAB(in_c + out_c, r, bias))
        self.dec_block3 = nn.Sequential(CAB(in_c + out_c + out_c, r, bias),
                                        CAB(in_c + out_c + out_c, r, bias))
        
        self.up2_1 = SkipUpSample(in_c, out_c)
        self.up3_2 = SkipUpSample(in_c + out_c, out_c)
        
        self.skip_con_1 = CAB(in_c, r, bias)
        self.skip_con_2 = CAB(in_c + out_c, r, bias)
        
    def forward(self, x):
        
        out1, out2, out3 = x
        dec3 = self.dec_block3(out3)
        
        res = self.up3_2(dec3, self.skip_con_2(out2))
        dec2 = self.dec_block2(res)
        
        res = self.up2_1(dec2, self.skip_con_1(out1))
        dec1 = self.dec_block1(res)
        
        return [dec1, dec2, dec3]
    
    
class ORB(nn.Module):
    
    '''
    ORB(Original Resolution Block) operates on original image resolution at final 
    stage to preserve the spatial details of the reconstructed images.
    
    
    Input ---> CAB(1) ---> CAB(2) ---...---> CAB(n) ---> Conv ---> + --> Out
           '                                                       ^
           '-------------------------------------------------------'
    
    '''
    
    def __init__(self, in_c, num_cab, r, bias):
        
        super().__init__()
        
        body = []
        body = [CAB(in_c, r, bias) for _ in range(num_cab)]
        body.append(nn.Conv2d(in_c, in_c, kernel_size = 3, padding = 1, bias = False))
        self.block = nn.Sequential(*body)
    
    def forward(self, x):
        
        out = self.block(x)
        out += x
        return out
    
    
class ORSNet(nn.Module):
    
    '''
    ORSNet is a collection of ORBs(Original Resolution Blocks) with csff.
    
    csff (Cross Stage Feature Fusion) at Stage 3:
    
     
      ORSNet---------------> + --> Out
                             |
       --------------------> + 
       '                     '
      Conv                  Conv
       ^                     ^
       '                     '               
    Encoder                Decoder
    
    '''
    
    def __init__(self, in_c, out_c, ors_out_c, num_cab, r, bias):
        
        super().__init__()
        
        self.orb = ORB(in_c + ors_out_c, num_cab, r, bias)
        self.up_1 = UpSample(in_c, out_c)
        self.up_2 = nn.Sequential(UpSample(in_c + out_c, out_c), UpSample(in_c, out_c))
        self.conv = nn.Conv2d(in_c, in_c + ors_out_c, kernel_size = 1, bias = bias)
        
    def forward(self, x, encoder_out, decoder_out):
        
        x = self.orb(x)
        x = x + self.conv(encoder_out[0]) + self.conv(decoder_out[0])
        
        x = self.orb(x)
        x = x + self.conv(self.up_1(encoder_out[1])) + self.conv(self.up_1(decoder_out[1]))
        
        x = self.orb(x)
        x = x + self.conv(self.up_2(encoder_out[2])) + self.conv(self.up_2(decoder_out[2]))
        
        return x


class MPRNet(nn.Module):

    def __init__(self, in_c = 40, out_c = 20, ors_out_c = 16, r = 4, num_cab = 8, bias = False):
        
        super().__init__()
        
        self.initial_block = nn.Sequential(nn.Conv2d(3, out_channels = in_c, kernel_size = 3, padding = 1, bias = bias),
                                           CAB(in_c, r, bias))
        
        self.stage1_enc = Encoder(in_c, out_c, r, csff = False, bias = bias)
        self.stage1_dec = Decoder(in_c, out_c, r, bias)
        
        self.stage2_enc = Encoder(in_c, out_c, r,csff = True, bias = bias)
        self.stage2_dec = Decoder(in_c, out_c, r, bias)
        
        self.stage3 = ORSNet(in_c, out_c, ors_out_c, num_cab, r, bias)
        
        self.sam1_2 = SAM(in_c, bias)
        self.sam2_3 = SAM(in_c, bias)
        
        self.conv_concat1_2 = nn.Conv2d(in_c + in_c, in_c, kernel_size = 3, padding = 1, bias = bias)
        self.conv_concat2_3 = nn.Conv2d(in_c + in_c, in_c + ors_out_c, kernel_size = 3, padding = 1, bias = bias)
        self.last_layer = nn.Conv2d(in_c + ors_out_c, 3, kernel_size = 3, padding = 1, bias = bias)
        
    def forward(self, img):
        
        H = img.shape[2]
        W = img.shape[3]
        
        '''Dividing Image into patches'''
        
        #Stage2 patches
        stage2_img_top = img[:, :, 0:int(H/2), :]
        stage2_img_bot = img[:, :, int(H/2):H, :]
        
        #Stage1 patches
        stage1_top_left_patch = stage2_img_top[:, :, :, 0:int(W/2)]
        stage1_top_right_patch = stage2_img_top[:, :, :, int(W/2):W]
        stage1_bot_left_patch = stage2_img_bot[:, :, :, 0:int(W/2)]
        stage1_bot_right_patch = stage2_img_bot[:, :, :, int(W/2):W]
        
        
        '''Stage 1'''
        
        #step1: Pass every patch to initial block(Conv + CAB):
        x1_top_left = self.initial_block(stage1_top_left_patch)
        x1_top_right = self.initial_block(stage1_top_right_patch)
        x1_bot_left = self.initial_block(stage1_bot_left_patch)
        x1_bot_right = self.initial_block(stage1_bot_right_patch)
        
        #step2: Pass all the features to Encoder
        feat_top_left = self.stage1_enc(x1_top_left)
        feat_top_right = self.stage1_enc(x1_top_right)
        feat_bot_left = self.stage1_enc(x1_bot_left)
        feat_bot_right = self.stage1_enc(x1_bot_right)
        
        #step3: Concat top and bottom features
        top_features = [torch.cat((i,j), dim = 3) for i,j in zip(feat_top_left, feat_top_right)]
        bot_features = [torch.cat((i,j), dim = 3) for i,j in zip(feat_bot_left, feat_bot_right)]
        
        #step4: Pass features through the Decoder
        out1_top = self.stage1_dec(top_features)
        out1_bot = self.stage1_dec(bot_features)
        
        #step5: Apply SAM(returns feature maps, restored image)
        sam_feats_top, stage1_top_img = self.sam1_2(out1_top[0], stage2_img_top)
        sam_feats_bot, stage1_bot_img = self.sam2_3(out1_bot[0], stage2_img_bot)
        
        #step6: concat the top and bottom part to get final stage1 output Image
        stage1_restored_img = torch.cat([stage1_top_img, stage1_bot_img], dim = 2)
        
        
        '''Stage 2'''
        
        x2_top = self.initial_block(stage2_img_top)
        x2_bot = self.initial_block(stage2_img_bot) 
        
        x2_top_cat = self.conv_concat1_2(torch.cat([x2_top, sam_feats_top], dim = 1))
        x2_bot_cat = self.conv_concat1_2(torch.cat([x2_bot, sam_feats_bot], dim = 1))
        
        feat_top = self.stage2_enc(x2_top_cat, top_features, out1_top)
        feat_bot = self.stage2_enc(x2_bot_cat, bot_features, out1_bot)
        
        stage2_features = [torch.cat((i,j), dim = 2) for i,j in zip(feat_top, feat_bot)]
        
        out2 = self.stage2_dec(stage2_features)
        
        sam_feats, stage2_restored_img = self.sam2_3(out2[0], img)
        
        
        '''Stage 3'''
        
        x3 = self.initial_block(img)
        x3_cat = self.conv_concat2_3(torch.cat([x3, sam_feats], dim = 1))
        
        stage3_features = self.stage3(x3_cat, stage2_features, out2)
        stage3_img = self.last_layer(stage3_features)
        stage3_resored_img = stage3_img + img
        
        return [stage3_resored_img, stage2_restored_img, stage1_restored_img]
    
      
def test():
    
    model = MPRNet()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    
    
if __name__ == '__main__':
    
    test()
