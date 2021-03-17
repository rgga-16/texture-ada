from helpers import crop
from PIL import Image
import numpy as np
import os
from args import args
from re import search


def main():

    input_dir = './inputs/style_images/masked'
    output_dir = './inputs/style_images/cropped'

    for file in os.listdir(input_dir):
        if search('cobonpue',file):
            input_path = os.path.join(input_dir,file)

            im = Image.open(input_path)
            arr = np.array(im)

            arr = crop.crop_spaces(arr)
            arr = crop.center_crop(arr,np.uint8(np.rint(0.05*arr.shape[0])))

            im_c = Image.fromarray(np.uint8(arr))
            
            output_file = '{}_cropped.png'.format(os.path.splitext(os.path.basename(input_path))[0])
            output_path = os.path.join(output_dir,output_file)

            # im_c.show()
            im_c.save(output_path)
    return








if __name__ == "__main__":
    main()
