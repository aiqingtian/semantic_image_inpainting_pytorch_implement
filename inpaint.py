from PIL import Image
import argparse
import os
import numpy as np
import torch
from datasets import MaskFaceDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from glob import glob
import pdb
from model import ModelInpaint
from dcgan import Generator, Discriminator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--generator',type=str,
                         help='Pretrained generator',
                         default='/GAN/models/gen_60000.pt' )
    parser.add_argument( '--discriminator',type=str,
                         help='Pretrained discriminator',
                         default='/GAN/models/dis_60000.pt' )
    parser.add_argument( '--imgSize',type=int,default=64 )
    parser.add_argument( '--batch_size',type=int,default=1)
    parser.add_argument( '--n_size',type=int,default=7,
                         help='size of neighborhood' )
    parser.add_argument( '--blend',action='store_true',default=True,
                         help="Blend predicted image to original image" )
    parser.add_argument( '--mask_csv',type=str,default='/GAN/mask/mask.csv',
                         help='path to the masked csv file' )
    parser.add_argument( '--mask_root',type=str,default='/GAN',
                         help='path to the masked root' )
    parser.add_argument( '--per_iter_step',type=int,
                         default=15000,help='number of steps per iteration' )
    args = parser.parse_args()
    return args

def saveimages( corrupted, completed, blended, index ):
    os.makedirs( 'completion', exist_ok=True )
    print('index is: ', index)
    save_image( corrupted,
                'completion/%d_corrupted.png' % index,
                nrow=corrupted.shape[ 0 ] // 5,
                normalize=True )
    save_image( completed,
                'completion/%d_completed.png' % index,
                nrow=completed.shape[ 0 ] // 5,
                normalize=True )
    save_image( blended,
                'completion/%d_blended.png' % index,
                nrow=corrupted.shape[ 0 ] // 5,
                normalize=True )

def main():
    args = parse_args()
    # Configure data loader
    celebA_dataset = MaskFaceDataset( args.mask_csv,
                                      args.mask_root,
                                      transform=transforms.Compose( [
                           transforms.Resize( (args.imgSize, args.imgSize) ),
                           transforms.ToTensor(),
                           transforms.Normalize( ( 0.5, 0.5, 0.5 ), ( 0.5, 0.5, 0.5 ) )
                       ] ))
    dataloader = torch.utils.data.DataLoader( celebA_dataset,batch_size=args.batch_size,shuffle=True, num_workers=6 )
    m = ModelInpaint( args )
    for i, ( imgs, masks ) in enumerate( dataloader ):
        masks = np.stack( ( masks, ) * 3, axis=1 )
        corrupted = imgs * torch.tensor( masks )
        completed, blended = m.inpaint( corrupted, masks )
        saveimages( corrupted, completed, blended, i )
        corrupted = blended

if __name__ == '__main__':
    main()
