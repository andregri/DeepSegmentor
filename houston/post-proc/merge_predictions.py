from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Merge the predictions of roadnet applying a bilinear blending")
parser.add_argument('--results_dir', type=str, default="./results/roadnet/test_latest/images", help="dir containing the predictions")
parser.add_argument('--image_number', type=int, default=1, help='the number of the patch (the first number of the triplet)')
args = parser.parse_args()

img_number = args.image_number
print('--------------------- IMAGE {} ---------------------'.format(img_number))

img_dir = Path.cwd() / Path('./datasets/Houston-Dataset/test_image')
assert img_dir.is_dir()

img_fnames  = sorted([f.as_posix() for f in img_dir.rglob('{}-*.png'.format(img_number))])
print("Number of images: {}".format(len(img_fnames)))

pred_dir = Path.cwd() / Path(args.results_dir)
assert pred_dir.is_dir()

#pred_fnames = sorted([f.as_posix() for f in pred_dir.rglob('{}-*label_pred.png'.format(img_number))])
pred_fnames = [pred_dir.as_posix() + '/' + f.split('/')[-1].replace('.png', '_label_pred.png') for f in img_fnames]
print("Number of predictions: {}".format(len(pred_fnames)))

im_shape = [2862, 2838]
H = 512
W = 512
step = 256

########################################################################################################################

# Create the mask
def bb_mask(H=512, W=512):

    mask = np.zeros((H,W), dtype=np.float32)

    H = int(H/2)
    W = int(W/2)

    for h in range(H):
        for w in range(W):
            # mask A
            mask[h+H,w+W] = (W-w) * (H-h) / (H * W)
            # mask B
            mask[h+H,w]   = w * (H - h) / (H * W)
            # mask C
            mask[h,  w]   = h * w / (H * W)
            # mask D
            mask[h,  w+W] = (W-w) * h / (H*W)

    #plt.figure(figsize=(10,10))
    #plt.imshow(mask, cmap='jet')
    #plt.show()
    #cv2.imshow('bb_mask', mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return mask

# Bilinear blending
def bilinearBlending(image):
    mask = np.zeros(image.shape, dtype=np.float32)
    mask[:,:,0] = bb_mask(image.shape[0], image.shape[1])
    mask[:,:,1] = bb_mask(image.shape[0], image.shape[1])
    mask[:,:,2] = bb_mask(image.shape[0], image.shape[1])
    norm_image = np.array(image/255.0, dtype=np.float32)
    norm_output = np.multiply(norm_image, mask)
    return norm_output

# Resize the predictions to the original size of the associated input image
def wasResized(img_fname, pred_fname, input_pred):
    img_shape  = cv2.imread(img_fname, cv2.IMREAD_COLOR).shape
    if img_shape != input_pred.shape:
        #print('- Resizing "{}"'.format(pred_fname))
        resized = cv2.resize(input_pred, (img_shape[1],img_shape[0]))
        #cv2.imwrite(pred_fname.replace('.png','_warped.png'), pred)
        #cv2.imwrite(pred_fname, resized)
        return resized
    else:
        return input_pred

# 
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


def imageMerge(output_shape, img_fnames, pred_fnames, save_path, sz=(H,W), step=256):
    assert os.path.isdir(save_path)

    # Create the empty output image
    output_img = np.zeros(tuple(output_shape) + (3,), dtype=np.float32)

    # Load cropping information
    new_h, new_w, offset_h, offset_w = crop_info(output_shape, (H,W), step)

    # Add each prediction to the merged image
    for (img_fname, pred_fname) in zip(img_fnames, pred_fnames):
        fname = pred_fname.replace('_label_pred.png','')
        fname = fname.split('/')[-1]
        row = int(fname.split('-')[-2])
        col = int(fname.split('-')[-1])

        # Load image
        pred = cv2.imread(pred_fname, cv2.IMREAD_COLOR)

        # Apply bilinear blending
        blended = bilinearBlending(pred)

        # Resize if necessary
        resized = wasResized(img_fname, pred_fname, blended)

        #cv2.imshow('Prediction: ' + fname, resized)
        #cv2.waitKey(0)

        h = row * step
        if row == new_h-1:
            h -= offset_h
        w = col * step
        if col == new_w-1:
            w -= offset_w
        output_img[h:h+H, w:w+W, :] += resized

        #cv2.imshow('Merged', output_img)
        #cv2.waitKey(10)
    #cv2.destroyAllWindows()
        

    #cv2.imshow('Merged', output_img)
    #cv2.resizeWindow('Merged', 800,800)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    output_png = np.array(output_img * 255, dtype=np.float32)
    output_fname = save_path + "Houston-{}-pred.png".format(img_number)
    cv2.imwrite(output_fname, output_png, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print('- Merged image was saved to {}'.format(output_fname))

imageMerge(im_shape, img_fnames, pred_fnames, '/tmp/')