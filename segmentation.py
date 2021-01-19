from dextr import segment

import args as a
import utils
from defaults import DEFAULTS as D

path = 'binary_mask.png'
args = a.parse_arguments()
# style_image = utils.load_image(args.style)
# mask = utils.load_image(path)

# mask_t = utils.image_to_tensor(mask,normalize=False)

# style = utils.load_image('chair-2.jpg')
# style_t = utils.image_to_tensor(style)

# output_t = style_t * mask_t

# output= utils.tensor_to_image(output_t)
# output = output.convert('RGBA')
# output.save('output.png','PNG')


mask = utils.load_image('uv_map.png',mode="L")

style = utils.load_image('output_chair-2_cropped.png',mode="RGBA")

output = style.copy()
output.putalpha(mask)
output.save('output.png','PNG')


# segment.segment_points(image_path = args.style,device=D.DEVICE())

