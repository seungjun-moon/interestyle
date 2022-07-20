import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

def landmark_dict(landmark):
    lm_dict={}
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

def add_masks(component_list,lm_dict,graphonomy_mask,soft_mask):
    points_lists=[]
    total_mask=get_mask(graphonomy_mask,[], soft_mask=soft_mask, expand_list=10 if soft_mask else 50)
    for i,component in enumerate(component_list):
        mask=get_mask(graphonomy_mask,[lm_dict[component]], soft_mask=soft_mask, expand_list=10 if soft_mask else 50)
        total_mask+=mask
        total_mask=np.where(total_mask>1, 1, total_mask)
        total_mask=np.where(total_mask<0, 0, total_mask) #Just clipping!!
    if not soft_mask:
        total_mask=total_mask.astype(np.uint8)
    return total_mask

def makeuppostprocess(image1_path,image2_path,landmark=None,graphonomy_mask=None,component_list=[]):
    if landmark is None:
        raise FileNotFoundError
    else:
        lm_dict=landmark_dict(landmark)

    im = cv2.imread(image1_path)
    im_makeup = cv2.imread(image2_path)

    #We need to divide the case when apply skin or not.

    if 'skin' in component_list:
        # skin_mask_soft=get_mask(cv2.imread(os.path.join(file_root,mask_name))/255, input_is_png=True, soft_mask=False)
        skin_mask_soft=get_mask(graphonomy_mask/255, input_is_png=True, soft_mask=True)
        skin_mask_hard=get_mask(graphonomy_mask/255, input_is_png=True, soft_mask=False)

        #Apply total makeup first.

        image_inter=(im_makeup * skin_mask_soft + im * (1-skin_mask_soft)).astype(np.uint8)
        image_inter=cv2.seamlessClone(im_makeup,image_inter,255*skin_mask_hard,(512,512),cv2.NORMAL_CLONE)

        # neg_component_list=[]
        # for component in ['eyebrow_left','eyebrow_right','eye_left','eye_right','mouth']:
        #     if component not in component_list: #not applying makeup
        #         neg_component_list.append('')

        neg_component_list = list(set(['eyebrow_left','eyebrow_right','eye_left','eye_right','mouth']) - set(component_list))
        neg_mask_soft=add_masks(neg_component_list,lm_dict,graphonomy_mask,soft_mask=True)
        neg_mask_hard=add_masks(neg_component_list,lm_dict,graphonomy_mask,soft_mask=False)
        image_output=(im * neg_mask_soft + image_inter * (1-neg_mask_soft)).astype(np.uint8)
        image_output=cv2.seamlessClone(im,image_output,255*neg_mask_hard,(512,512),cv2.NORMAL_CLONE)

    else:
        mask_soft=add_masks(component_list,lm_dict,graphonomy_mask,soft_mask=True)
        mask_hard=add_masks(component_list,lm_dict,graphonomy_mask,soft_mask=False)

        image_output=(im_makeup * mask_soft + im * (1-mask_soft)).astype(np.uint8)
        image_output=cv2.seamlessClone(im_makeup,image_output,255*mask_hard,(512,512),cv2.NORMAL_CLONE)

    return image_output

def makeuppostprocess(image1_path,image2_path,landmark=None, graphonomy_mask=None,component_list=[]):
    if graphonomy_mask is None or landmark is None:
        raise FileNotFoundError

    im = cv2.imread(image1_path)
    im_makeup = cv2.imread(image2_path)

    lm=landmark
    lm_dict=landmark_dict(landmark)

    skin_mask_soft=get_mask(graphonomy_mask/255, input_is_png=True, soft_mask=True)
    skin_mask_hard=get_mask(graphonomy_mask/255, input_is_png=True, soft_mask=False)

    #Apply total makeup first.

    image_inter=(im_makeup * skin_mask_soft + im * (1-skin_mask_soft)).astype(np.uint8)
    image_inter=cv2.seamlessClone(im_makeup,image_inter,255*skin_mask_hard,(512,512),cv2.NORMAL_CLONE)

    neg_component_list = list(set(['eyebrow_left','eyebrow_right','eye_left','eye_right','mouth','nose']) - set(component_list))
    neg_mask_soft=add_masks(neg_component_list,lm_dict,graphonomy_mask,soft_mask=True)
    neg_mask_hard=add_masks(neg_component_list,lm_dict,graphonomy_mask,soft_mask=False)
    image_output=(im * neg_mask_soft + image_inter * (1-neg_mask_soft)).astype(np.uint8)
    image_output=cv2.seamlessClone(im,image_output,255*neg_mask_hard,(512,512),cv2.NORMAL_CLONE)

    return image_output