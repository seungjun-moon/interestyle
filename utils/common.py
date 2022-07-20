import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	n_outputs = len(log_hooks[0]['output_face']) if type(log_hooks[0]['output_face']) == list else 1
	fig = plt.figure(figsize=(6 + (n_outputs * 2), 4 * display_count))
	gs = fig.add_gridspec(display_count, (2 + n_outputs))
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		vis_faces_iterative(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def vis_faces_iterative(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']), float(hooks_dict['diff_target'])))
	for idx, output_idx in enumerate(range(len(hooks_dict['output_face']) - 1, -1, -1)):
		output_image, similarity = hooks_dict['output_face'][output_idx]
		fig.add_subplot(gs[i, 2 + idx])
		plt.imshow(output_image)
		plt.title('Output {}\n Target Sim={:.2f}'.format(output_idx, float(similarity)))

def get_concat_h(image_list, w, h):
    image_size = image_list[0].width
    dst = Image.new('RGB', (image_size*(w+1), image_size*(h+1)))
    for i,image in enumerate(image_list):
        dst.paste(image, (((i+1) % (w+1))*image_size,((i+1)//(w+1))*image_size))
    return dst

def blend(image_list):
    for i,name in enumerate(image_list):
        if i==0:
            if os.path.isfile('./latents/'+name+'_searched.npy'):
                latent = torch.from_numpy(np.load('./latents/'+name+'_searched.npy'))
            else:
                latent = torch.from_numpy(np.load('./latents/'+name+'.npy'))
        else:
            if os.path.isfile('./latents/'+name+'_searched.npy'):
                latent += torch.from_numpy(np.load('./latents/'+name+'_searched.npy'))
            else:
                latent += torch.from_numpy(np.load('./latents/'+name+'.npy'))
    mean = latent/len(image_list)
    return mean

def stylemixing(image_list, style_name):
    for i,name in enumerate(image_list):
        if i==0:
            latent = torch.from_numpy(np.load('./latents/'+name+'.npy'))
        else:
            new_latent = torch.from_numpy(np.load('./latents/'+name+'.npy'))
    if style_name=='coarse':
        start=0
        end=3
    elif style_name=='middle':
        start=4
        end=7
    else:
        start=8
        end=17
    for i in range(start,end+1):
        latent[0][i]=new_latent[0][i]
        
    return latent

def makeupblend(image_list,thresholds,latent_path='./latents',latent_path2='./latents'):
    id_name = image_list[0]
    makeup_name = image_list[1]
    id_latent = torch.from_numpy(np.load(os.path.join(latent_path,id_name+'.npy')))
    makeup_latent = torch.from_numpy(np.load(os.path.join(latent_path2,makeup_name+'.npy')))

    latent = torch.empty(size=id_latent.shape)
    threshold_list = thresholds.split(',')
    s_threshold = int(threshold_list[0]) #start
    e_threshold = int(threshold_list[1]) #end
    diff = e_threshold - s_threshold
    for i in range(18):
        if i < s_threshold: #0,1,2, ... , s_t-1
            latent[0][i]=id_latent[0][i]
        elif i >= s_threshold and i < e_threshold: #s_t, ... e_t-1
            latent[0][i]= ((e_threshold-i)*id_latent[0][i]+ (i-s_threshold)*makeup_latent[0][i])/(e_threshold-s_threshold)
        else:
            latent[0][i]=makeup_latent[0][i]

    ##### Nonetheless, I have to do this.
    id_latent2 = torch.from_numpy(np.load(os.path.join(latent_path2,id_name+'.npy')))

    indexes_9 =[32, 380, 157, 474, 9, 184, 359, 261, 64, 178, 143, 476, 138, 284, 189, 187, 209, 340, 215, 186, 37, 228]
    for idx in indexes_9:
        latent[0][9][idx]=makeup_latent[0][9][idx]

    return latent

def latentblend(latent1,latent2,thresholds):
    id_latent = latent1
    makeup_latent = latent2

    latent = torch.empty(size=id_latent.shape)
    threshold_list = thresholds.split(',')
    s_threshold = int(threshold_list[0]) #start
    e_threshold = int(threshold_list[1]) #end
    diff = e_threshold - s_threshold
    for i in range(18):
        if i < s_threshold:
            latent[0][i]=id_latent[0][i]
        elif i >= s_threshold and i < e_threshold:
            latent[0][i]= ((i-s_threshold)*id_latent[0][i]+ (e_threshold-i)*makeup_latent[0][i])/(e_threshold-s_threshold)
        else:
            latent[0][i]=makeup_latent[0][i]

    return latent

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

def imagepostprocess(image1_path,image2_path,landmark=None, NOSE=True): #landmark of image1
    if landmark is None:
        from ffhq_dataset.landmarks_detector import LandmarksDetector
        from align_images import unpack_bz2
        from keras.utils import get_file

        LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))

        landmarks_detector = LandmarksDetector(landmarks_model_path)
        for landmark in landmarks_detector.get_landmarks(image1_path):
            landmark = np.array(landmark)
            print(landmark.shape)
            print('./landmark/'+image1_path.split('/')[-1])
            '''
            This will occur error when image name contains '/'.
            In my case, image name is always a format 'frontal_{}_01.png'.
            However, if you want to change the name of image, be careful of this!!
            '''
            np.save('./landmark/'+image1_path.split('/')[-1], landmark)

    im = cv2.imread(image1_path) #frontal face
    im_new = cv2.imread(image2_path)
    lm=landmark

    im = cv2.imread(image1_path)
    im_new = cv2.imread(image2_path)


    ### Histogram Matching??? ###

    '''
    from skimage import exposure
    multi = True if im.shape[-1] > 1 else False
    im_hm = exposure.match_histograms(im_new, im[100:900,200:800], multichannel=multi)
    im_new = (im_new/3*2 + im_hm/3).astype(np.uint8)
    cv2.imwrite('mix_hm.png', im_hm)
    '''
    from skimage import exposure
    multi = True if im.shape[-1] > 1 else False
    im_hm = exposure.match_histograms(im, im_new[400:900,250:750], multichannel=multi)
    im=(im/3*2 + im_hm/3).astype(np.uint8)
    cv2.imwrite('mix_hm.png', im)

    if NOSE:
        mask=get_mask(im,[lm[36 : 42],lm[42 : 48], np.stack((lm[27],lm[31],lm[35]),axis=0),lm[48:60]], expand_list=[10,10,30,10], ksize=71, sigma=70)
        seamless_mask=get_mask(im,[lm[36 : 42],lm[42 : 48],np.stack((lm[27],lm[31],lm[35])),lm[48:60]], expand_list=[50,50,60,20], soft_mask=False)
    else:
        mask=get_mask(im,[lm[36 : 42],lm[42 : 48], lm[48:60]], expand_list=[10,10,25])
        seamless_mask=get_mask(im,[lm[36 : 42],lm[42 : 48],lm[48 : 60]], expand_list=[30,30,35], soft_mask=False) #frontal hair --> Narrow Mask

    image_new=(im_new * mask + im * (1-mask)).astype(np.uint8)
    cv2.imwrite('mix_intermediate.png', image_new)

    image_new=cv2.seamlessClone(im_new,image_new,255*seamless_mask,(512,512),cv2.NORMAL_CLONE)
    image_new=image_new.astype(np.uint8)

    return(image_new)

def makeuppostprocess(image1_path,image2_path,landmark=None): #landmark of image1
    if landmark is None:
        from ffhq_dataset.landmarks_detector import LandmarksDetector
        from align_images import unpack_bz2
        from keras.utils import get_file

        LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))

        landmarks_detector = LandmarksDetector(landmarks_model_path)
        for landmark in landmarks_detector.get_landmarks(image1_path):
            landmark = np.array(landmark)
            '''
            This will occur error when image name contains '/'.
            In my case, image name is always a format 'frontal_{}_01.png'.
            However, if you want to change the name of image, be careful of this!!
            '''
            np.save('./landmark/'+image1_path.split('/')[-1], landmark)

    im = cv2.imread(image1_path) #frontal face
    im_new = cv2.imread(image2_path)
    lm=landmark

    im = cv2.imread(image1_path)
    im_new = cv2.imread(image2_path)

    mask=get_mask(im,[lm[36 : 42],lm[17 : 22],lm[42 : 48],lm[22 : 27],lm[48:60]], expand_list=10)
    seamless_mask_with_mix=get_mask(im,[lm[36 : 42],lm[42 : 48]], expand_list=50, soft_mask=False)
    seamless_mask_with_normal=get_mask(im,[lm[17 : 22],lm[22 : 27],lm[48:60]], expand_list=50, soft_mask=False)

    image_new=(im_new * mask + im * (1-mask)).astype(np.uint8)

    image_new=cv2.seamlessClone(im_new,image_new,255*seamless_mask_with_mix,(512,512),cv2.MIXED_CLONE)
    image_new=cv2.seamlessClone(im_new,image_new,255*seamless_mask_with_normal,(512,512),cv2.NORMAL_CLONE)

    image_new=image_new.astype(np.uint8)

    return image_new
