
import argparse
from defaults import DEFAULTS as D
import datetime
import os 


def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description='Texture Transfer Algorithm')
    parser.add_argument('--uv_style_pairs',type=str, 
                        help='Path to .json file of uv map and style image pairings.')
    parser.add_argument('--style_dir', type=str, default=None,
                        help='Path to the style images directory to retrieve textures from')
    parser.add_argument('--uv_dir', type=str, default=None,
                        help='Path to the uv maps directory of UV maps to transfer textures onto')                    
    parser.add_argument('--output_dir', type=str, default='./outputs/output_images/[{}]'.format(D.DATE()),
                        help='Path to the output directory')
    parser.add_argument('--epochs', type=int, default=D.EPOCHS(),
                        help='Number of epochs to optimize')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate of texture transfer network')
    parser.add_argument('--style_size', type=int, default=D.IMSIZE.get(),
                        help='Size of input style images')
    parser.add_argument('--uv_train_sizes', type=int, nargs="*",default=[64,128,254],
                        help='Sizes of input uv map images during training')
    parser.add_argument('--uv_test_sizes', type=int, nargs="*",default=[512,768,1024],
                        help='Sizes of input uv maps images during testing')
    parser.add_argument('--output_size', type=int, default=D.IMSIZE.get(),
                        help='Size of output textured uv map')
    parser.add_argument('--style_weight', type=float, default=1e6,
                        help='Style loss weight value')
    parser.add_argument('--num_batch_chkpts', type=int,default=5,
                        help='Number of checkpoints to print batch training losses')
    parser.add_argument('--num_epoch_chkpts', type=int,default=5,
                        help='Number of checkpoints to print epoch training losses')
    parser.add_argument('--num_points', type=int,default=2048,
                        help='Number of points to sample for pointcloud (structure transfer)')
    parser.add_argument('--multiprocess', type=bool,default=False,
                        help='To use multiple cpu cores for the dataloader or not')

    return parser.parse_args()

global args
args = parse_arguments()