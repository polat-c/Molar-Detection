import argparse
## lel

import numpy as np
import torch

from solver import DetectionSolver
from utils import str2bool

def detect_ROI(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = DetectionSolver(args)

    net.load_checkpoint(args.ckpt_name)
    net.detect(args.dir_images, args.csv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy-AE')

    ## detection arguments
    parser.add_argument('--dir_images', default='data/annotated/patches/images_nmasks', type=str, help='directory of images')
    #######################

    parser.add_argument('--model', default='DetectionConvNet', type=str, help='current models [ConvNet, ResNetXIn]')
    parser.add_argument('--molar_guarantee', default=False, type=str2bool, help='if True only use images with molars')
    ## extra arguments
    parser.add_argument('--predictor', default='Regressor', type=str,
                        help='Regressor or Classifier dependent on the construction of the detection problem')  # change 'Classifier' to train Unet
    #######################
    parser.add_argument('--window_size', default=256, type=int, help='image width and height (should be the same)')
    parser.add_argument('--res_size',default=18, type=int, help='if model = resnet, size can be [18,34,50,101,152]' )
    parser.add_argument('--pretrain',default=False, type=bool, help='if model = resnet, choose if pretrained or not')

    parser.add_argument('--cuda', default=False, type=str2bool, help='enable cuda')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--evaluation', default=True, type=str2bool, help='enable evaluation')

    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--num_workers', default=1, type=int, help='dataloader num_workers')
    parser.add_argument('--lrate_decay', type=float, default=0.8, help='Learning rate decay.')
    parser.add_argument('--patience', type=int, default=5, help='Patience on learning rate decay.')
    parser.add_argument('--weight_decay', type=float, default = 0.01, help='lambda for L2 loss')

    parser.add_argument('--dset_dir', default='data/annotated/patches/images_nmasks', type=str, help='directory of data')
    parser.add_argument('--csv_file', default='data/annotated/annotations/all/cv1/test_annotations.csv', type=str, help='path to .csv annotations')
    parser.add_argument('--test_portion', default=0.2, type=float, help='test set size')
    parser.add_argument('--predictor_type', default='simple', type=str, help='type of predictor for model evaluation')

    parser.add_argument('--eval_step', default=20, type=int,
                        help='number of iterations after which full model evaluation occurs')
    parser.add_argument('--gather_step', default=20, type=int,
                        help='number of iterations after which data is gathered for tensorboard')
    parser.add_argument('--save_step', default=20, type=int,
                        help='number of iterations after which a checkpoint is saved')
    parser.add_argument('--viz_step', default=20, type=int,
                        help='number of iterations after which images are displayed in tensorboard')

    parser.add_argument('--ckpt_dir', default='dcheckpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='best', type=str,
                        help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--output_dir', default='doutputs', type=str, help='output directory')
    parser.add_argument('--exp_name', default='main', type=str, help='name of the experiment')

    parser.add_argument('--log_dir', default='logdir', type=str, help='directory of logs')

    args = parser.parse_args()

    detect_ROI(args)