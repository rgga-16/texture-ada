import numpy as np
import cv2


def blend_images(content_img,stylized_img,mask):

    if(content_img.shape[:2] != stylized_img.shape[:2]):
        stylized_img = cv2.resize(stylized_img, tuple(reversed(content_img.shape[:2])), interpolation = cv2.INTER_NEAREST)

    if(content_img.shape[:2] != mask.shape[:2]):
        mask = cv2.resize(mask, tuple(reversed(content_img.shape[:2])), interpolation = cv2.INTER_NEAREST)

    final_image = stylized_img * mask + content_img * (1-mask)

    return final_image

def crop_with_mask(image,mask):
    if(image.shape[:2] != mask.shape[:2]):
        mask = cv2.resize(mask, tuple(reversed(image.shape[:2])), interpolation = cv2.INTER_NEAREST)

    cropped_image = image * mask
    return cropped_image
