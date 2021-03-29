import torch
from torch import nn
from network import *
from utils import *
import itertools
from torch.utils.data import DataLoader
from dataset import *
from torchvision import transforms
from transforms import *
import time
import edsr
import neptune
from pytorch_fid import fid_score
import numpy as np
class CinCGAN():
    def __init__(self, gpu, ngpus_per_node, args):
        # Device configuration
        
        if ngpus_per_node>1:
            # DDP init
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='tcp://127.0.0.1:3456',
                world_size=ngpus_per_node,
                rank=gpu)
            
        torch.cuda.set_device(gpu)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("GPU : ", gpu)
        self.lr = args.inner_lr
        self.decay_step = args.decay_step
        self.batch_size = args.bsize//ngpus_per_node
        self.test_batch_size = args.test_bsize
        print(f"GPU : {gpu}, BatchSize : {self.batch_size}")
        self.iteration = args.iteration
        self.eps = args.eps
        self.w1 = args.w1
        self.w2 = args.w2
        self.w3 = args.w3

        self.lr_outer = args.outer_lr
        
        self.gamma0 = args.gamma0
        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2
        self.gamma3 = args.gamma3
        self.gamma4 = args.gamma4

        self.train_s_path = args.train_s_path
        self.train_t_path = args.train_t_path
        self.test_s_path = args.test_s_path
        self.test_t_path = args.test_t_path
        self.fid_s_path = self.test_s_path
        self.gpu = gpu
        self.ngpus_per_node = ngpus_per_node
        self.num_workers = args.num_workers

        self.step = 0

        self.resume = args.resume
        self.resume_iter = args.resume_iter
        self.inner_ckpt_path = args.inner_ckpt_path
        self.neptune = args.neptune
        self.params = vars(args)

        
        
        self.save_freq = args.save_freq

        self.scale_factor = args.scale_factor
        self.skip_inner = args.skip_inner
        self.use_fid = args.use_fid
        self.fid_save_path = args.fid_save_path
        self.fid_s_path = args.fid_s_path
        self.fid_t_path = args.fid_t_path
        self.fid_bsize = args.fid_bsize
        self.fid_freq = args.fid_freq
        self.fid_mean = None
        self.fid_cov = None
        self.use_psnr = args.use_psnr
        self.psnr_freq = args.psnr_freq
        self.ckpt_save_path = args.ckpt_save_path
        if os.path.isdir('/mnt/nas/workspace/sr1/target_fid_statistics'):
            self.fid_mean = np.load(os.path.join('/mnt/nas/workspace/sr1/target_fid_statistics', 'm2.npy'))
            self.fid_cov = np.load(os.path.join('/mnt/nas/workspace/sr1/target_fid_statistics', 's2.npy'))
            

        self.inception = None
        self.best_psnr = args.best_psnr

        self.imsize_y = 128
        assert self.imsize_y % self.scale_factor == 0, 'invalid scale factor'
        self.imsize_x = 128//self.scale_factor
        self.max_hw = args.max_hw
        print(f"NEPTUNE : {self.neptune}")
        print(f"gamma0 : {self.gamma0}")
        print(f"gamma1 : {self.gamma1}")
        print(f"gamma2 : {self.gamma2}")
        print(f"gamma3 : {self.gamma3}")
        print(f"gamma4 : {self.gamma4}")
        print(f"train_s_path : {self.train_s_path}")
        print(f"train_t_path : {self.train_t_path}")
        
        print(f"test_s_path : {self.test_s_path}")
        print(f"test_t_path : {self.test_t_path}")
        print(f"scale factor : {self.scale_factor}")

    def build_model(self):
        
        
        """ Define Inner Cycle Models """

        
        self.G1_forward = ResnetGenerator(scale_factor = self.scale_factor).to(self.device)
        self.G1_backward = ResnetGenerator(scale_factor = self.scale_factor).to(self.device)
        self.D1_forward = Discriminator(scale_factor = self.scale_factor).to(self.device)
        self.D1_backward = Discriminator(scale_factor = self.scale_factor).to(self.device)

        """ Define Outer Cycle Models """

        self.EDSR = edsr.EDSR(gpu = self.gpu, scale_factor=self.scale_factor).cuda(self.gpu) # should be changed
        self.G3 = ResnetGenerator(dsple=True, scale_factor = self.scale_factor).cuda(self.gpu)
        self.D2 = Discriminator(is_inner=False, scale_factor = self.scale_factor).cuda(self.gpu)


        """ Define Loss """
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.TV_loss= TVLoss().to(self.device)
        
        self.PSNR = PSNR(self.gpu)

        if self.ngpus_per_node>1:
        
            self.G1_forward = torch.nn.parallel.DistributedDataParallel(self.G1_forward, device_ids=[self.gpu], broadcast_buffers=False)
            self.G1_backward = torch.nn.parallel.DistributedDataParallel(self.G1_backward, device_ids=[self.gpu], broadcast_buffers=False)
            self.D1_forward = torch.nn.parallel.DistributedDataParallel(self.D1_forward, device_ids=[self.gpu], broadcast_buffers=False)
            self.D1_backward = torch.nn.parallel.DistributedDataParallel(self.D1_backward, device_ids=[self.gpu], broadcast_buffers=False)
            
            """ Define Loss """
            self.MSE_loss = nn.MSELoss().cuda(self.gpu)
            self.TV_loss= TVLoss().cuda(self.gpu)
        
        




        

        """ Inner Optimizers """
        self.G_optim = torch.optim.Adam(itertools.chain(self.G1_forward.parameters(), self.G1_backward.parameters()),lr=self.lr, betas=(0.5,0.999), eps = self.eps)
        self.D_optim = torch.optim.Adam(itertools.chain(self.D1_forward.parameters(), self.D1_backward.parameters()),lr=self.lr, betas=(0.5,0.999), eps = self.eps)
        
        """ Outer Optimizers """
        self.G_outer_optim = torch.optim.Adam(itertools.chain(self.G3.parameters(), self.G1_forward.parameters(), self.EDSR.parameters()),lr=self.lr_outer, betas=(0.5,0.999), eps = self.eps)
        self.D2_optim = torch.optim.Adam(self.D2.parameters(),lr=self.lr_outer, betas=(0.5,0.999), eps = self.eps)
        """ Data loader """

        train_transform_s = transforms.Compose([
            Random90Rot(),
            transforms.RandomCrop((self.imsize_x,self.imsize_x)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        train_transform_t = transforms.Compose([
            Random90Rot(),
            transforms.RandomCrop((self.imsize_y,self.imsize_y)),
            transforms.Resize((self.imsize_x,self.imsize_x), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        

        test_transform = transforms.Compose([
            Crop(self.max_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        fid_s_transform = transforms.Compose([
            transforms.CenterCrop((self.imsize_x, self.imsize_x)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        


        self.train_s_dataset = DatasetFolder(self.train_s_path, 0, transform=train_transform_s)
        self.train_t_dataset = DatasetFolder(self.train_t_path, 1, transform=None, return_two_img = True, big_imsize=self.imsize_y, scale_factor = self.scale_factor)
        self.train_t_big_dataset = DatasetFolder(self.train_t_path, 1, transform=None, return_two_img = True)
        
        self.test_s_dataset = DatasetFolder(self.test_s_path, 0, transform=test_transform)
        self.test_t_dataset = DatasetFolder(self.test_t_path, 1, transform=test_transform)

        self.fid_s_dataset = DatasetFolder(self.fid_s_path, 0, transform=fid_s_transform)
        
        if self.ngpus_per_node>1:
            train_s_sampler = torch.utils.data.distributed.DistributedSampler(self.train_s_dataset, num_replicas=self.ngpus_per_node, rank=self.gpu)
            train_t_sampler = torch.utils.data.distributed.DistributedSampler(self.train_t_dataset, num_replicas=self.ngpus_per_node, rank=self.gpu)
            self.train_s_loader = DataLoader(self.train_s_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=train_s_sampler)
            self.train_t_loader = DataLoader(self.train_t_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=train_t_sampler)
        else:
            self.train_s_loader = DataLoader(self.train_s_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            self.train_t_big_loader = DataLoader(self.train_t_big_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            self.train_t_loader = DataLoader(self.train_t_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            self.fid_s_loader = DataLoader(self.fid_s_dataset, batch_size=self.fid_bsize, shuffle=False, num_workers=self.num_workers)
            

        
        self.test_s_loader = DataLoader(self.test_s_dataset, batch_size=self.test_batch_size, shuffle=False)
        self.test_t_loader = DataLoader(self.test_t_dataset, batch_size=self.test_batch_size, shuffle=False)
        
        
    
    def train(self, inner=True):
        psnr = 0
        psnr_inner = 0
        check_g3 = 0
        fid_inner = 0
        fid_outer = 0
        self.inner = inner
        # torch.autograd.set_detect_anomaly(True)
        for model in [self.G1_forward, self.G1_backward, self.D1_forward, self.D1_backward, self.EDSR, self.D2, self.G3]:
            model.train()
        if self.skip_inner:
            self.G1_forward.eval()
        start_iter = 1
        print(f"NEPTUNE : {self.neptune}")
        if self.neptune:
            neptune.init(project_qualified_name='ml.swlee/CinCGAN',
                        api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYmVhNmFlOGMtZDRmZS00NzIyLWJkYzgtNTcyZTk0ZTM5YzM1In0=')
            neptune.create_experiment(params = self.params)
        if self.resume:
            start_iter = self.resume_iter
            if inner:
                self.load(path_inner = self.inner_ckpt_path)

        # Validation data
        x_test, _ = iter(self.test_s_loader).next()
        x_test = x_test.to(self.device)

        y_test_big, _ = iter(self.test_t_loader).next()
        # if self.use_fid == False:
        #     assert x_test.size()[0] == y_test_big.size()[0] == 1
        y_test = transforms.Resize(x_test.size()[-2:], interpolation=Image.BICUBIC)(tensor2pil(y_test_big[0])) # To resize, tensor should be converted to pillow image.
        y_test = transforms.ToTensor()(y_test)
        y_test = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(y_test)
        y_test = torch.unsqueeze(y_test,0)
        y_test = y_test.to(self.device)
        y_test_big = y_test_big.cuda(self.gpu)


                
        if self.use_psnr: base_d = self.PSNR(denorm(x_test[0]), denorm(y_test[0]))
        if self.ngpus_per_node>1:
            if self.gpu==0:
                x_test, _ = iter(self.test_s_loader).next()
                x_test = x_test.cuda(self.gpu)

                y_test_big, _ = iter(self.test_t_loader).next()
                assert x_test.size()[0] == y_test_big.size()[0] == 1
                y_test = transforms.Resize(x_test.size()[-2:], interpolation=Image.BICUBIC)(tensor2pil(y_test_big[0])) # To resize, tensor should be converted to pillow image.
                y_test = torch.unsqueeze(transforms.ToTensor()(y_test),0)
                y_test = y_test.cuda(self.gpu)
                
                
                if self.use_psnr: base_d = self.PSNR(x_test*0.5 + 0.5, y_test*0.5 + 0.5)
        
        

        if self.skip_inner:
            freeze_model(self.G1_forward)
            print("----------------------------Model Freezed")
       

        for step in range(start_iter, self.iteration+1):
            self.step = step
            t = time.time()
            if step%self.decay_step==0:
                n = step//self.decay_step
                self.G_optim.param_groups[0]['lr'] = self.lr/2**n # param_groups = [{'amsgrad': False, 'betas': (...), 'eps': 1e-08, 'lr': 0.0002, 'params': [...], 'weight_decay': 0}]
                self.D_optim.param_groups[0]['lr'] = self.lr/2**n
                self.G_outer_optim.param_groups[0]['lr'] = self.lr_outer/2**n
                self.D2_optim.param_groups[0]['lr'] = self.lr_outer/2**n
                
            
            try:
                x, label_x = train_s_iter.next()
            except:
                train_s_iter = iter(self.train_s_loader)
                x, label_x = train_s_iter.next()
            
            try:
                y_big, y, label_y = train_t_iter.next()
                
            except:
                train_t_iter = iter(self.train_t_loader)
                y_big, y, label_y = train_t_iter.next()
                
            
            if self.ngpus_per_node>1:
                x, y = x.cuda(self.gpu), y.cuda(self.gpu)
                


            x, y = x.to(self.device), y.to(self.device)
            

            y_big = y_big.cuda(self.gpu)
            print(f"x.shape : {x.size()}, y.shape : {y.size()}, y_big.shape : {y_big.size()}")




            if self.gpu==0: print(f"Data loading : {time.time()-t:.3f}")
            t = time.time()
            # Update discriminator
            self.D_optim.zero_grad()
            if self.skip_inner==False:
                

                fake_forward = self.G1_forward(x)
                fake_backward = self.G1_backward(y)
                print(f"fake_forward.shape : {fake_forward.size()} fake_backward.shape : {fake_backward.size()} ")
                # D1 output for real images
                dis1_forward_output_real = self.D1_forward(y)
                dis1_backward_output_real = self.D1_backward(x)
                
                # D1 output for fake images
                dis1_forward_output_fake = self.D1_forward(fake_forward)
                dis1_backward_output_fake = self.D1_backward(fake_backward)
                print(f"dis1_forward_output_fake.shape : {dis1_forward_output_fake.size()} dis1_backward_output_fake.shape : {dis1_backward_output_fake.size()} ")
                # D1 prediction for real images
                dis1_forward_predicted_real = torch.mean(dis1_forward_output_real, dim = [1,2,3])
                dis1_backward_predicted_real = torch.mean(dis1_backward_output_real, dim = [1,2,3])

                # D1 prediction for fake images
                dis1_forward_predicted_fake = torch.mean(dis1_forward_output_fake, dim = [1,2,3])
                dis1_backward_predicted_fake = torch.mean(dis1_backward_output_fake, dim = [1,2,3])

            
                
                # Discriminator loss
                dis1_forward_ad_loss = self.MSE_loss(dis1_forward_predicted_fake, torch.zeros_like(dis1_forward_predicted_fake).cuda(self.gpu)) + self.MSE_loss(dis1_forward_predicted_real, torch.ones_like(dis1_forward_predicted_real).cuda(self.gpu))
                dis1_backward_ad_loss = self.MSE_loss(dis1_backward_predicted_fake, torch.zeros_like(dis1_backward_predicted_fake).cuda(self.gpu)) + self.MSE_loss(dis1_backward_predicted_real, torch.ones_like(dis1_backward_predicted_real).cuda(self.gpu))

                dis1_loss = dis1_forward_ad_loss + dis1_backward_ad_loss

                dis1_loss.backward()
                self.D_optim.step()

                
                
                # Update Generator
                self.G_optim.zero_grad()

                fake_forward = self.G1_forward(x)
                fake_backward = self.G1_backward(y)

                # D1 output for fake images
                dis1_forward_output_fake = self.D1_forward(fake_forward)
                dis1_backward_output_fake = self.D1_backward(fake_backward)

                # D1 prediction for fake images
                dis1_forward_predicted_fake = torch.mean(dis1_forward_output_fake, dim = [1,2,3])
                dis1_backward_predicted_fake = torch.mean(dis1_backward_output_fake, dim = [1,2,3])
                
                # Adversarial loss
                g1_forward_ad_loss = self.MSE_loss(dis1_forward_predicted_fake, torch.ones_like(dis1_forward_predicted_fake).cuda(self.gpu))
                g1_backward_ad_loss = self.MSE_loss(dis1_backward_predicted_fake, torch.ones_like(dis1_backward_predicted_fake).cuda(self.gpu))

                # Cycle consistency loss
                g1_forward_cycle_loss = self.MSE_loss(self.G1_backward(fake_forward), x)
                g1_backward_cycle_loss = self.MSE_loss(self.G1_forward(fake_backward), y)

                # Identity loss
                g1_forward_identity_loss = self.MSE_loss(self.G1_forward(y), y)
                g1_backward_identity_loss = self.MSE_loss(self.G1_backward(x), x)

                # Total variation loss
                g1_forward_tv_loss = self.TV_loss(fake_forward)
                
                if inner==False:
                    self.w2 = 1

                g1_forward_loss = g1_forward_ad_loss + self.w1*g1_forward_cycle_loss + self.w2*g1_forward_identity_loss + self.w3*g1_forward_tv_loss
                g1_backward_loss = g1_backward_ad_loss + self.w1*g1_backward_cycle_loss + self.w2*g1_backward_identity_loss

                g1_loss = g1_forward_loss + g1_backward_loss

                g1_loss.backward()
                self.G_optim.step()
                
                if inner==True:
                    print(f"endloop : {time.time()-t:.3f}")
                    t = time.time()

                    print(f"step : {step}  d_loss : {dis1_loss}  g_loss : {g1_loss}  lr : {self.D_optim.param_groups[0]['lr']}")
            # get inner psnr
            
            self.G1_forward.eval()
            
            with torch.no_grad():
                img_t = self.G1_forward(x_test)[0]
                if self.use_psnr and step%self.psnr_freq==0:
                    #psnr = self.PSNR(img_t*0.5 + 0.5, y_test[0]*0.5 + 0.5)
                    psnr_inner = self.psnr_mean(inner=True)
                    print(f"inner psnr : {psnr_inner}")
                    if inner:
                        if psnr_inner > self.best_psnr:
                            self.best_psnr = psnr_inner
                            self.save(is_outer = False, fname = f'inner_{step}_{psnr_inner.item():0.4f}.pt')
            if inner==True and self.use_fid==True:
                if step%self.fid_freq==0:
                    fid_inner = self.fid()

            if self.skip_inner == False:
                self.G1_forward.train()
            if self.neptune:
                if self.use_psnr: neptune.log_metric('inner_psnr', psnr_inner)
                if self.use_fid: neptune.log_metric('inner_fid', fid_inner)

            if inner==True:
                if step%50==0:

                    tensor_imsave(img_t, f'/mnt/nas/workspace/sr1/imgs', f'img{step}.png')
                if self.skip_inner == False: self.G1_forward.train()
            
            if inner==True:
                if step%self.save_freq == 0:
                    self.save(is_outer=False)
                continue
            
            """ outer update """
            
            # Update D2
            self.D2_optim.zero_grad()

            fake_forward_inner = self.G1_forward(x)
            fake_forward_outer = self.EDSR(fake_forward_inner)

            
            d2_output_fake = self.D2(fake_forward_outer)
            d2_output_real = self.D2(y_big*0.5 + 0.5)

            d2_predicted_fake = torch.mean(d2_output_fake, dim = [1,2,3])
            d2_predicted_real = torch.mean(d2_output_real, dim = [1,2,3])

            # Check D2
            print(f"D2 prediction Avg -- fake : {torch.mean(d2_predicted_fake):0.4f}  real : {torch.mean(d2_predicted_real):0.4f}")

            d2_ad_loss = self.MSE_loss(d2_predicted_fake, torch.zeros_like(d2_predicted_fake).cuda(self.gpu)) + self.MSE_loss(d2_predicted_real, torch.ones_like(d2_predicted_real))

            d2_ad_loss.backward()
            self.D2_optim.step()

            # Update G1, EDSR, G3
            self.G_outer_optim.zero_grad()
            fake_forward_inner = self.G1_forward(x)
            fake_forward_outer = self.EDSR(fake_forward_inner)

            
            d2_output_fake = self.D2(fake_forward_outer)
            d2_predicted_fake = torch.mean(d2_output_fake, dim = [1,2,3])


            # Adversarial loss
            edsr_ad_loss = self.MSE_loss(d2_predicted_fake, torch.ones_like(d2_predicted_fake))

            # Cycle consistency loss
            edsr_cycle_loss = self.MSE_loss(self.G3(fake_forward_outer), x*0.5+0.5)
            
            # Total variation loss
            edsr_tv_loss = self.TV_loss(fake_forward_outer)

            # Identity loss
            edsr_identity_loss = self.MSE_loss(self.EDSR(y), y_big*0.5 + 0.5)


            # G3 loss
            g3_loss = self.MSE_loss(self.G3(y_big*0.5 + 0.5), y*0.5 + 0.5)

            g_outer_loss = self.gamma0*edsr_ad_loss + self.gamma1*edsr_cycle_loss + self.gamma2*edsr_identity_loss + self.gamma3*edsr_tv_loss

        
            
            # Check G3
            if self.use_psnr and step%self.psnr_freq==0:
                with torch.no_grad():
                    temp = self.G3(y_test_big.cuda(self.gpu)*0.5 + 0.5)
                    check_g3 = self.PSNR(temp, x_test*0.5 + 0.5)

                print(f"Check_g3_PSNR : {check_g3:0.4f}")
            
            g_outer_loss.backward()
            self.G_outer_optim.step()
            
            if self.gpu==0: print(f"step : {step}")
            if self.use_fid==True:
                if step%self.fid_freq==0:
                    fid_outer = self.fid()

            if self.use_psnr and step%self.psnr_freq==0:
                with torch.no_grad():
                    # psnr = self.PSNR(out, y_test_big*0.5 + 0.5)
                    psnr = self.psnr_mean(inner=False)
                    print("----outer psnr : ", psnr.item())
                    if inner==False:
                        if psnr > self.best_psnr:
                            self.best_psnr = psnr
                            self.save(fname = f'outer_{step}_{psnr.item():0.4f}.pt')
                            print("----------------Best ckpt saved ------------------")
                    if self.skip_inner == False: self.G1_forward.train()
            if self.neptune:
                if self.use_psnr:
                    neptune.log_metric('outer_psnr', psnr)
                    neptune.log_metric('outer_check_g3_PSNR', check_g3)
                if self.use_fid:
                    neptune.log_metric('outer fid', fid_outer)
                neptune.log_metric("outer_identity loss", edsr_identity_loss)
                neptune.log_metric('outer_cycle_loss', edsr_cycle_loss)
                neptune.log_metric('outer_tv_loss', edsr_tv_loss)
                neptune.log_metric('outer_edsr_adversarial loss', edsr_ad_loss)
                neptune.log_metric('outer_D2_adversarial loss', d2_ad_loss)
                
            with torch.no_grad():
                if step%50==0:
                    self.G1_forward.eval()
                    try:
                        out = self.G1_forward(x_test)    
                        out = self.EDSR(out)
                        tensor_imsave(out[0], '/mnt/nas/workspace/sr1/finetuning', f'{step}.png', denormalization=False)
                    except:
                        None
            if step%self.save_freq==0:
                self.save()
            
        self.save(is_outer=True)
        print("------Checkpoint saved-----")
    def save(self, is_outer=True, fname = 'default'):
        if fname == 'default':
            fname = f'{self.step}.pt'
        check_folder(os.path.join(self.ckpt_save_path, 'ckpt'))
        params = {}
        params['G1_forward'] = self.G1_forward.state_dict()
        params['G1_backward'] = self.G1_backward.state_dict()
        params['D1_forward'] = self.D1_forward.state_dict()
        params['D1_backward'] = self.D1_backward.state_dict()
        
        torch.save(params, os.path.join(self.ckpt_save_path, 'ckpt', fname))
        if is_outer==True:
            check_folder(os.path.join(self.ckpt_save_path, 'ckpt_edsr'))
            params_edsr = {}
            params_edsr['EDSR'] = self.EDSR.state_dict()
            torch.save(params_edsr, os.path.join(self.ckpt_save_path,'ckpt_edsr', fname))
        
            
    def load(self, path_inner, inner=True, path_outer = ''):
        ckpt = torch.load(path_inner)
        
        self.G1_forward.load_state_dict(ckpt['G1_forward'])
        self.G1_backward.load_state_dict(ckpt['G1_backward'])
        self.D1_forward.load_state_dict(ckpt['D1_forward'])
        self.D1_backward.load_state_dict(ckpt['D1_backward'])

        if inner==False:
            assert len(path_outer)>0
            ckpt = torch.load(path_outer)
            if list(ckpt.keys())[0] == 'EDSR':
                ckpt = ckpt['EDSR']
            self.EDSR.load_state_dict(ckpt)
        print("CKPT Loaded")
    def test(self, inner=True, idx=0):
        # inner -- true : test inner cycle, false : test outer cycle
        # idx -- Load n-th image in test dataset folder
        test_s_iter = iter(self.test_s_loader)
        if self.use_psnr: test_t_iter = iter(self.test_t_loader)
        for i in range(idx+1):    
            x_test, x_name = test_s_iter.next()
        x_test = x_test.cuda(self.gpu)
        
        self.G1_forward.eval()
        
        
        if self.use_psnr:
            for i in range(idx+1):
                y_test, _ = test_t_iter.next()
        
            y_test = y_test.cuda(self.gpu)    
        
            
        with torch.no_grad():
            fake_inner = self.G1_forward(x_test)
            
        
        if inner==True :
            for img, fname in zip(fake_inner, x_name):
                tensor_imsave(img, '/mnt/nas/workspace/sr1/test', fname)
            if self.use_psnr:
                psnr = self.PSNR(denorm(y_test),fake_inner)
                
                return psnr
            else:
                return None
        
        self.EDSR.synchronize_norm = True
        self.EDSR.eval()
        with torch.no_grad():
            fake_outer = self.EDSR(fake_inner)
        for img, fname in zip(fake_outer, x_name):
            tensor_imsave(fake_outer[0], '/mnt/nas/workspace/sr1/test', fname, denormalization=False)
        if self.use_psnr:
            #psnr = -10*torch.log10(torch.mean((denorm(y_test[0])-fake_outer[0])**2))
            psnr = self.PSNR(denorm(y_test), fake_outer)
        
            return psnr
        else:
            return None
    def psnr_mean(self, inner, num=10):
        # Derive mean of psnr of whole test data
        # num : int, number of samples
        test_s_iter = iter(self.test_s_loader)    
        test_t_iter = iter(self.test_t_loader)
        num = np.minimum(num, len(self.test_s_loader))
        self.G1_forward.eval()
        self.EDSR.eval()
        psnr_mean = 0
        i=0
        with torch.no_grad():
            while True:
                try:
                    x, _ = test_s_iter.next()
                except:
                    break                
                try:
                    y_big, _ = test_t_iter.next()
                except:
                    break
                y = transforms.Resize(x.size()[-2:], interpolation=Image.BICUBIC)(tensor2pil(y_big[0])) # To resize, tensor should be converted to pillow image.
                y = transforms.ToTensor()(y)
                y = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(y)
                y = torch.unsqueeze(y,0)
                y = y.cuda(self.gpu)

                x, y_big = x.cuda(self.gpu), y_big.cuda(self.gpu)
                # fake_inner = self.G1_forward(x)
                out = self.G1_forward(x)
                
                if inner==False:
                    out = self.EDSR(out)
                    psnr = self.PSNR(y_big*0.5 + 0.5, out)
                if inner:
                    out = out * 0.5 + 0.5
                    psnr = self.PSNR(y*0.5 + 0.5, out)
                psnr_mean += psnr/num
                print(f"i : {i}, psnr : {psnr:0.4f}")
                i+=1
                if i==num:
                    break
        return psnr_mean
    
    def fid(self):
        
        # Generate imgs
        self.val_sr_save()
        # Get fid score
        
        result, self.fid_mean, self.fid_cov, self.inception = fid_score.calculate_fid_given_paths([self.fid_save_path, self.fid_t_path], 
									128, self.device, 2048, self.fid_mean, self.fid_cov, self.inception)
        return result
    
    def val_sr_save(self, n = 20000):
        iters = n // self.fid_bsize
        if iters>len(self.fid_s_loader):
            iters = len(self.fid_s_loader)
        fid_iter = iter(self.fid_s_loader)
        num = 1
        check_folder(self.fid_save_path)
        for i in range(iters):
            with torch.no_grad():
                x_fid, _ = fid_iter.next()
                x_fid = x_fid.cuda(self.gpu)
                
                out = self.G1_forward(x_fid)
                if self.inner==False:
                    out = self.EDSR(out)

                for img in out:
                    d = True if self.inner else False
                    tensor_imsave(img, self.fid_save_path, f'{num}.png', denormalization = d, prt = False)
                    num += 1