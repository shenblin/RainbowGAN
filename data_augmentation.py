import os
import glob
import cv2
import numpy as np


imgs_path = 'datasets/HE_segmentation/Ovarian/ovarian_RGB_exp'
# imgs_path = 'datasets/HE_segmentation/Ovarian/ovarian_GT_exp'


def flip_augmentation():
    im_files = sorted(list(glob.glob(os.path.join(imgs_path, '*.tif'))))
    num_imgs = len(im_files)
    for i_img, im_file0 in enumerate(im_files):
        im_file = os.path.split(im_file0)[-1].split('_')[0]
        img = cv2.imread(im_file0, -1)

        flip_list = [1, 0, -1]  # 1: horizontal flip; 0: vertical flip; -1: h&v flip
        num_flip = len(flip_list)
        for idx, flip in enumerate(flip_list):
            img_flip = cv2.flip(img, flip)
            # cv2.imshow('img', img_flip)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            save_img_path = os.path.join(imgs_path, (im_file + '_' + str(num_imgs+1+ num_flip*i_img + idx) + '.tif'))
            cv2.imwrite(save_img_path, img_flip)
            print(save_img_path)


def rotation_augmentation():
    im_files = sorted(list(glob.glob(os.path.join(imgs_path, '*.tif'))))
    num_imgs = len(im_files)
    for i_img, im_file0 in enumerate(im_files):
        im_file = os.path.split(im_file0)[-1].split('_')[0]
        img = cv2.imread(im_file0, -1)

        num_rotation = 1
        for idx in range(num_rotation):
            if idx == 0:
                img_rotation = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                img_rotation = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            save_img_path = os.path.join(imgs_path, (im_file + '_' + str(num_imgs+1+ num_rotation*i_img + idx) + '.tif'))
            cv2.imwrite(save_img_path, img_rotation)
            print(save_img_path)


def shift_augmentation():
    im_files = sorted(list(glob.glob(os.path.join(imgs_path, '*.tif'))))
    num_imgs = len(im_files)
    k = num_imgs+1
    for i_img, im_file0 in enumerate(im_files):
        im_file = os.path.split(im_file0)[-1].split('_')[0]
        img = cv2.imread(im_file0, -1)
        h, w = img.shape[0:2]

        # shift_list = np.linspace(136, 408, 3)  # pixel shift
        shift_list = np.linspace(362, 362, 1)  # pixel shift


        for idx, pixel_shift in enumerate(shift_list):
            pixel_shift = int(pixel_shift)
            img_shift = np.vstack((img[(h - pixel_shift):, :], img[:(h - pixel_shift), :]))
            save_img_path = os.path.join(imgs_path, (im_file + '_' + str(k) + '.tif'))
            cv2.imwrite(save_img_path, img_shift)
            print(save_img_path)
            k += 1

        for idy, pixel_shift in enumerate(shift_list):
            pixel_shift = int(pixel_shift)
            img_shift = np.hstack((img[:, (w - pixel_shift):], img[:, :(w - pixel_shift)]))
            save_img_path = os.path.join(imgs_path, (im_file + '_' + str(k) + '.tif'))
            cv2.imwrite(save_img_path, img_shift)
            print(save_img_path)
            k += 1


flip_augmentation()

# rotation_augmentation()

shift_augmentation()

print('done!')
