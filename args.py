
import argparse
from defaults import DEFAULTS as D


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Texture Transfer Algorithm')

    parser.add_argument('--mesh', type=str, default=D.MESH_PATH(),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--image', type=str, default=D.STYLE_PATH(),
                        help='Path to the style image to transfer texture from')
    parser.add_argument('--output_path', type=str, default='outputs',
                        help='Path to the output directory')
    parser.add_argument('--epochs', type=int, default=D.EPOCHS(),
                        help='Number of epochs to optimize')
    parser.add_argument('--camera_distance', type=float, default=2.732,
                        help='Distance from camera to object center')
    parser.add_argument('--elevation', type=float, default=0,
                        help='Camera elevation')
    parser.add_argument('--texture_size', type=int, default=4,
                        help='Dimension of texture')

    return parser.parse_args()