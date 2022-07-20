import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

name='746_01'

def landmark_dict(landmark):
    lm_dict={}
    lm_dict['chin'] = landmark[0:17]
    lm_dict['eyebrow_left']  = landmark[17 : 22]
    lm_dict['eyebrow_right']  = landmark[22 : 27]
    lm_dict['nose']  = np.stack((landmark[27],landmark[31],landmark[35]),axis=0)
    lm_dict['eye_left']  = landmark[36 : 42]
    lm_dict['eye_right']  = landmark[42 : 48]
    lm_dict['mouth']  = landmark[48 : 60]
    return lm_dict

def get_mask(im, points_lists=None, expand_list=0, soft_mask=True, input_is_png=False, ksize=51, sigma=10):
    if not input_is_png:
        mask = np.zeros(im.shape[:2], np.uint8)
        if type(expand_list)==int:
            expand_list=[expand_list for i in range(len(points_lists))]
        for num,points in enumerate(points_lists):
            expand=expand_list[num]
            new_points=np.empty(points.shape)
            mean = np.mean(points, axis=0)
            for i, point in enumerate(points):
                distance=np.linalg.norm(point-mean)
                new_points[i]=(point+(expand*(point-mean)/distance)//1)
        new_points=new_points.astype(np.int32)
        mask=cv2.fillPoly(mask, [new_points],(1,1,1))
        mask=np.stack((mask,mask,mask), axis=2)
    else:
        mask = im
        mask=np.where(mask>0.9, 1, mask)
        mask=np.where(mask<0.9, 0, mask).astype(np.uint8)
    if soft_mask:
        mask=mask.astype(np.float)
        mask = cv2.GaussianBlur(mask, (ksize, ksize),sigma,sigma,cv2.BORDER_DEFAULT)

    mask[0:10]=1.0
    mask[-10:-1]=1.0

    return mask

if __name__ == '__main__':
    landmark = np.load('./landmark/{}.png.npy'.format(name))
    lm_dict = landmark_dict(landmark)
    image = cv2.imread('./bases/{}.png'.format(name))
    print(lm_dict['chin'])
    mask = get_mask(im=image, points_lists=[lm_dict['chin']], expand_list=0, soft_mask=False)

    image = image * mask
    cv2.imwrite('mask.png',image)
