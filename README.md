# CinCGAN-pytorch

Pytorch implementation of ["Unsupervised Image Super-Resolution using Cycle-in-Cycle Generative Adversarial Networks"](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Yuan_Unsupervised_Image_Super-Resolution_CVPR_2018_paper.html), CVPRW 2018

# **Experiment Results**


## NTIRE2020 Track 1 - Average PSNR of Validation Set(x4)

|Name|PSNR(RGB)|
|---|---|
|Bicubic|24.21|
|EDSR|23.93|
|CinCGAN|24.92|

# Training Details

After training the inner cycle for 400K iterations with the default setting, I froze the inner cycle and fine-tuned the outer cycle for a few hundred iterations. While training the outer cycle, gamma0, gamma2, and outer_lr was set to 0.1, 150, and 1e-5, respectively.

# Checkpoints

I used EDSR implementation from [here](https://github.com/sanghyun-son/EDSR-PyTorch). 

NTIRE2020 x4 checkpoint : [https://drive.google.com/file/d/1ctTPy0dxHd5PgGvDc6rtJ8-8wNIjx86w/view?usp=sharing](https://drive.google.com/file/d/1ctTPy0dxHd5PgGvDc6rtJ8-8wNIjx86w/view?usp=sharing)

# Start Training

## Train Inner Cycle

```python
python3 main.py --phase train_inner --train_s_path "YOUR PATH" --train_t_path "YOUR PATH" --test_s_path "YOUR PATH" --test_t_path "YOUR PATH"
```

## Train Outer Cycle

```python
python3 main.py --inner_ckpt_path ".../.../inner_ckpt.pt" --outer_ckpt_path ".../.../EDSR_x4.pt"--phase train_outer --gamma0 0.1 --gamma2 150 --outer_lr 1e-5 --skip_inner True --train_s_path "YOUR PATH" --train_t_path "YOUR PATH" --test_s_path "YOUR PATH" --test_t_path "YOUR PATH"
```
