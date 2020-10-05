from pathlib import Path
import os
import numpy as np
import cv2

img_dir = Path('/home/andrea/Downloads/join-images/test_image')
pred_dir = Path('/home/andrea/Downloads/join-images/test_latest/images')

########################################################################################################################

# Resize the predictions to the original size of the associated input image

img_fnames  = sorted([f.as_posix() for f in img_dir.rglob('*.png')])
pred_fnames = [f.replace('/test_image', '/test_latest/images').replace('.png', '_label_pred.png') for f in img_fnames]

for img_fname, pred_fname in zip(img_fnames, pred_fnames):
    img_shape  = cv2.imread(img_fname, cv2.IMREAD_COLOR).shape
    pred = cv2.imread(pred_fname, cv2.IMREAD_COLOR)
    if img_shape != pred.shape:
        print('- Resizing "{}"'.format(pred_fname))
        cv2.imwrite(pred_fname.replace('.png','_warped.png'), pred)
        resized = cv2.resize(pred, (img_shape[1],img_shape[0]))
        cv2.imwrite(pred_fname, resized)

########################################################################################################################
im_shape = [2862, 2838]

info_file  = [f for f in img_dir.rglob('*.info')]
with open(info_file[0].as_posix()) as info:
    line = info.readlines()[0]
    infos = line.split()
    crop_info = {}
    crop_info['new_h']    = infos[0]
    crop_info['new_w']    = infos[1]
    crop_info['offset_h'] = infos[2]
    crop_info['offset_w'] = infos[3]

im_shape = [2862, 2838]
H = 512
W = 512
step = 256
def crop_info(im_shape, sz=(H,W), step=step):
    new_h = im_shape[0] / step
    offset_h = im_shape[0] % step
    if offset_h > 0:
        new_h += 1
        offset_h = step - offset_h
    new_w = im_shape[1] / step
    offset_w = im_shape[1] % step
    if offset_w > 0:
        new_w += 1
        offset_w = step - offset_w
    return int(new_h), int(new_w), offset_h, offset_w


new_h, new_w, offset_h, offset_w = crop_info(im_shape)


def imageCrop(im_file, save_path):
    assert os.path.isdir(save_path)
    # get the image index 
    fname = im_file.split('/')[-2]
    # load image and calculate cropping information
    im = cv2.imread(im_file, IMG_READ_MODE)
    s = im.shape
    new_h, new_w, offset_h, offset_w = crop_info(s)
    # save cropping information
    fp = open(os.path.join(save_path, '{}.info'.format(fname)), 'w')
    fp.write(str(new_h)+' '+str(new_w)+' '+str(offset_h)+' '+str(offset_w))
    fp.close()
    print("cropping info: ", new_h, new_w, offset_h, offset_w)
    # crop and save
    h, w = 0, 0
    for i in range(new_h):
        h = i * step
        if i == new_h-1:
            h -= offset_h
        for j in range(new_w):
            w = j * step
            if j == new_w-1:
                w -= offset_w
            im_roi = im[h:h+H, w:w+W, :]
            cv2.imwrite(os.path.join(save_path, 
                "{}-{}-{}.png".format(fname, i, j)), 
                im_roi, PNG_SAVE_MODE)

def imageMerge(output_shape, pred_fnames, save_path, sz=(H,W), step=256):
    assert os.path.isdir(save_path)

    # Create the empty output image
    output_img = np.zeros(tuple(output_shape) + (3,), dtype=np.uint8)

    # Load cropping information
    new_h, new_w, offset_h, offset_w = crop_info(output_shape, (H,W), step)

    # Add each prediction to the merged image
    for pred_fname in pred_fnames:
        fname = pred_fname.replace('_label_pred.png','')
        fname = fname.split('/')[-1]
        row = int(fname.split('-')[-2])
        col = int(fname.split('-')[-1])

        pred = cv2.imread(pred_fname, cv2.IMREAD_COLOR)

        cv2.imshow('Prediction: ' + fname, pred)
        cv2.waitKey(0)

        h = row * step
        if row == new_h-1:
            h -= offset_h
        w = col * step
        if col == new_w-1:
            w -= offset_w
        output_img[h:h+H, w:w+W, :] = pred

        cv2.imshow('Merged', output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

imageMerge(im_shape, pred_fnames, '/tmp/')