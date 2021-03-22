
import argparse
from defaults import DEFAULTS as D


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Texture Transfer Algorithm')
    parser.add_argument('--mesh', type=str, default=D.MESH_PATH(),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--style', type=str, default=D.STYLE_PATH(),
                        help='Path to the style image to transfer texture from')
    parser.add_argument('--style_dir', type=str, default='./inputs/style_images/masked',
                        help='Path to the style images dir to transfer textures from')
    parser.add_argument('--content_dir', type=str, default='./inputs/uv_maps/office chair/unwrap',
                        help='Path to the content images dir to transfer textures onto')                    
    parser.add_argument('--texture', type=str, default=None,
                        help='Path to the texture image to transfer texture from')
    parser.add_argument('--content', type=str, default=None,
                        help='Path to the content image to transfer texture onto')
    parser.add_argument('--output_dir', type=str, default='./outputs/output_images',
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
    parser.add_argument('--foreground_weight', type=float, default=1e2,
                        help='Foreground MSE loss weight value')
    parser.add_argument('--content_weight', type=float, default=1e2,
                        help='Content loss weight value')

    return parser.parse_args()

global args
args = parse_arguments()