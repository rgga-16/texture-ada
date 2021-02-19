
import argparse
from defaults import DEFAULTS as D


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Texture Transfer Algorithm')
    parser.add_argument('--mesh', type=str, default=D.MESH_PATH(),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--style', type=str, default=D.STYLE_PATH(),
                        help='Path to the style image to transfer texture from')
    parser.add_argument('--style_dir', type=str, default='./inputs/style_images',
                        help='Path to the style images dir to transfer textures from')
    parser.add_argument('--content_dir', type=str, default='./inputs/uv_maps',
                        help='Path to the content images dir to transfer textures onto')                    
    parser.add_argument('--texture', type=str, default=None,
                        help='Path to the texture image to transfer texture from')
    parser.add_argument('--content', type=str, default=None,
                        help='Path to the content image to transfer texture onto')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Path to the output directory')
    parser.add_argument('--epochs', type=int, default=D.EPOCHS(),
                        help='Number of epochs to optimize')
    parser.add_argument('--imsize', type=int, default=D.IMSIZE.get(),
                        help='Size to generated image')

    return parser.parse_args()

global args
args = parse_arguments()