# Image-deraining
PyTorch Implementation of [MPRNet](https://arxiv.org/abs/2011.04566).
<br>
Deep learning model to remove rain from the images.

## Results:

This model achieves 26 dB PSNR on [Rain100H Dataset](https://drive.google.com/drive/folders/17RKnYBq0rlbABS73u_WcLjcZyxYHPY-T?usp=sharing).

Input :point_right: restored image :point_right: ground-truth

![1](https://github.com/Vrushank264/Image-deraining/blob/main/Results/6.png)
![2](https://github.com/Vrushank264/Image-deraining/blob/main/Results/7.png)
![3](https://github.com/Vrushank264/Image-deraining/blob/main/Results/8.png)
![4](https://github.com/Vrushank264/Image-deraining/blob/main/Results/3.png)
![5](https://github.com/Vrushank264/Image-deraining/blob/main/Results/5.png)

More examples are in the `Results` directory.

### Details:

1. This model is trained on [Rain Dataset](https://drive.google.com/drive/folders/1Hnnlc5kI0v9_BtfMytC2LR5VpLAFZtVe).
2. The model is optimized using two losses: `Charbonnier Loss and Laplacian Edge Loss`.
3. All the training images were resized to `64x64` to fit them into a single GPU.
4. Learning rate is set to `2e-4` and it is decayed overtime to `1e-6`.
5. Trained model is available in `Pretrained model` directory.
 
