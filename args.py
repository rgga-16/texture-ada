
import argparse
from defaults import DEFAULTS as D


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Texture Transfer Algorithm')

    parser.add_argument('--mesh', type=str, default=D.MESH_PATH(),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--style', type=str, default=D.STYLE_PATH(),
                        help='Path to the style image to transfer texture from')
    parser.add_argument('--texture', type=str,
                        help='Path to the texture image to transfer texture from')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Path to the output directory')
    parser.add_argument('--epochs', type=int, default=D.EPOCHS(),
                        help='Number of epochs to optimize')
    parser.add_argument('--camera_distance', type=float, default=D.CAM_DISTANCE(),
                        help='Distance from camera to object center')
    parser.add_argument('--texture_size', type=int, default=D.TEXTURE_SIZE(),
                        help='Dimension of texture')
    parser.add_argument('--imsize', type=int, default=D.IMSIZE.get(),
                        help='Size to generated image')

    return parser.parse_args()