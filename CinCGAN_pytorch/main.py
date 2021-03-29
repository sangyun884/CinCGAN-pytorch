from CinCGAN import CinCGAN
import torch
import os
import argparse

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of CinCGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gpu', type=str, default='0', help='GPU idx')
    parser.add_argument('--inner_lr', type=float, default=2*1e-4, help='initial inner lr')
    parser.add_argument('--outer_lr', type=float, default=1e-4, help='initial outer lr')
    parser.add_argument('--decay_step', type=int, default=40000, help='decay step')
    parser.add_argument('--bsize', type=int, default=16, help='Batch size')
    parser.add_argument('--test_bsize', type=int, default=1, help='Test Batch size')
    parser.add_argument('--iteration', type=int, default=400000, help='iteration')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam eps')
    parser.add_argument('--w1', type=float, default=10.0, help='w1')
    parser.add_argument('--w2', type=float, default=5.0, help='w2')
    parser.add_argument('--w3', type=float, default=0.5, help='w3')
    parser.add_argument('--gamma0', type=float, default=1.0, help='gamma0')
    parser.add_argument('--gamma1', type=float, default=10.0, help='gamma1')
    parser.add_argument('--gamma2', type=float, default=5.0, help='gamma2')
    parser.add_argument('--gamma3', type=float, default=2.0, help='gamma3')
    parser.add_argument('--gamma4', type=float, default=0.0, help='gamma4')
    
    parser.add_argument('--train_s_path', type=str, default='/mnt/nas/data/track1/Corrupted-tr-x/bicubic', help='train source dataset path')
    parser.add_argument('--train_t_path', type=str, default='/mnt/nas/data/track1/Corrupted-tr-y', help='train target dataset path')
    parser.add_argument('--test_s_path', type=str, default='/mnt/nas/data/track1/Corrupted-va-x', help='test source dataset path')
    parser.add_argument('--test_t_path', type=str, default='/mnt/nas/data/track1/DIV2K_valid_HR', help='test target dataset path')
    parser.add_argument('--ngpus_per_node', type=int, default=1, help='ngpus per node')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--resume', type=bool, default=False, help='resume or not')
    parser.add_argument('--resume_iter', type=int, default=1, help='resume iteration')
    
    parser.add_argument('--neptune', type=bool, default=False, help='to use neptune or not')
    parser.add_argument('--save_freq', type=int, default=1000, help='checkpoint save frequency')
    parser.add_argument('--phase', type=str, default='train_inner', help='train_inner / train_outer / test_inner / test_outer')
    parser.add_argument('--scale_factor', type=int, default=4, help='scale factor : 2 / 4')
    parser.add_argument('--inner_ckpt_path', type=str, default='/mnt/nas/workspace/sr1/400000.pt', help='Inner cycle checkpoint file path for fine tuning')
    parser.add_argument('--outer_ckpt_path', type=str, default='/mnt/nas/workspace/sr1/EDSR_x4.pt', help='Outer cycle checkpoint file path for fine tuning')
    parser.add_argument('--skip_inner', type=bool, default=False, help='Not update inner cycle')
    parser.add_argument('--use_fid', type=bool, default=False, help='Whether to use fid or not')
    parser.add_argument('--fid_save_path', type=str, default='/mnt/nas/workspace/sr1/fid', help='Fid save path')
    parser.add_argument('--fid_s_path', type=str, default='/mnt/nas/data/sat_data/test/k3a/imgs', help='Fid source path')
    parser.add_argument('--fid_t_path', type=str, default='/mnt/nas/data/sat_data/test/wv3_cropped', help='Fid target path')
    parser.add_argument('--fid_bsize', type=int, default=128, help='FID batch size')
    parser.add_argument('--fid_freq', type=int, default=1000, help='FID frequency')
    parser.add_argument('--best_psnr', type=float, default=25.0, help='Best psnr')
    parser.add_argument('--ckpt_save_path', type=str, default='/mnt/nas/workspace/sr1', help='ckpt save path')
    parser.add_argument('--use_psnr', type=bool, default=True, help='Whether to use psnr or not')
    parser.add_argument('--psnr_freq', type=int, default=100, help='PSNR frequency')
    parser.add_argument('--max_hw', type=int, default=1000, help='Max test input image size')

  
    return check_args(parser.parse_args())

def check_args(args):
    try:
        assert args.iteration >= 1
    except:
        print('number of iteration must be larger than or equal to one')

    try:
        assert args.ngpus_per_node == 1
    except:
        print('Multi-gpu is not supported yet')
    return args
    
    
    
    
    
    
    
    
    
def main_worker_inner(gpu, ngpus_per_node, args):


    gan = CinCGAN(gpu, ngpus_per_node, args)
    gan.test_batch_size = args.test_bsize
    gan.build_model()
    gan.train(inner=True)
def main_worker_outer(gpu, ngpus_per_node, args):


    gan = CinCGAN(gpu, ngpus_per_node, args)
    gan.test_batch_size = args.test_bsize
    gan.build_model()
    gan.load(args.inner_ckpt_path, inner= False, path_outer=args.outer_ckpt_path)
    gan.train(inner=False)

    

def main():
    args = parse_args()


    gpu = args.gpu
    ngpus_per_node = args.ngpus_per_node
    
    #os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    phase =args.phase
    if phase == 'train_inner' or phase == 'train_outer':
        
        main_worker = main_worker_inner if phase=='train_inner' else main_worker_outer
        # ngpus_per_node = torch.cuda.device_count()
        world_size = ngpus_per_node
        print("Available gpu num : ", ngpus_per_node)

        if ngpus_per_node >1:
            torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, ))
        else:
            main_worker(int(gpu), 1, args)


    elif phase == 'test_inner' or phase == 'test_outer':
        
        print('----------Test mode----------')
        gan = CinCGAN(int(gpu), 1, args)
        gan.build_model()
        gan.load(args.inner_ckpt_path, inner= phase=='test_inner', path_outer=args.outer_ckpt_path)
        for i in range(18):
            gan.test(inner= phase=='test_inner', idx=i)
        i = 0
        # print(gan.psnr_mean(inner=False, num=800))
        # result = gan.psnr_mean()
        # print(f"Mean PSNR : {result}")
if __name__ == '__main__':
    main()