from PIL import Image
import numpy as np
import os


'''
Removes blank spaces outside the object in image
'''
def crop_spaces(arr):
    alpha = arr[...,3]
    nrows = len(alpha)
    ncols = len(alpha[0])
    t_idx=0
    b_idx=0
    l_idx=0
    r_idx=0
    for t in range(0,nrows):
        if np.sum(alpha[t,:]) > 0:
            t_idx=t 
            break 
    for b in range(nrows-1,-1,-1):
        if np.sum(alpha[b,:]) > 0:
            b_idx=b 
            break 
    for l in range(0,ncols):
        thing = alpha[:,l]
        if np.sum(alpha[:,l]) > 0:
            l_idx=l 
            break 
    for r in range(ncols-1,-1,-1):
        if np.sum(alpha[:,r]) > 0:
            r_idx=r
            break     

    arr_cropped =   arr[t_idx:b_idx+1,l_idx:r_idx+1,:]  
    return arr_cropped

'''
Center crops an image_arr equally from all sides by an amount
Args:
image_arr - Numpy array to be cropped
n_crop - Integer no. of rows and cols to crop by

Returns:
Center-cropped array
'''
def center_crop(image_arr,n_crop):
    nrows,ncols,_ = image_arr.shape
    return image_arr[n_crop:nrows-n_crop+1,
                    n_crop:ncols-n_crop+1,  
                    :]

def main():

    input_path = args.style
    im = Image.open(input_path)
    arr = np.array(im)

    arr = crop_spaces(arr)
    arr = center_crop(arr,np.uint8(np.rint(0.05*arr.shape[0])))

    im_c = Image.fromarray(np.uint8(arr))

    output_dir = args.output_dir
    output_file = '{}.png'.format(os.path.splitext(os.path.basename(input_path))[0])
    output_path = os.path.join(output_dir,output_file)

    im_c.show()
    im_c.save(output_path)
    return








if __name__ == "__main__":
    main()
