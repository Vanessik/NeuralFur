from glob import glob
from tqdm import tqdm
import os
from argparse import ArgumentParser
from PIL import Image
import torch
import numpy as np
import cv2
import pickle as pkl
import math


def main(args):
    data_path = args.data_path
    if os.path.exists(f'{data_path}/iqa_filtered_names.pkl'):
        img_names = pkl.load(open(f'{data_path}/iqa_filtered_names.pkl', 'rb'))
    else:
        img_names = os.listdir(f'{data_path}/images')

    os.makedirs(f'{data_path}/images_2', exist_ok=True)
    os.makedirs(f'{data_path}/images_4', exist_ok=True)
    os.makedirs(f'{data_path}/masks_2/hair', exist_ok=True)
    os.makedirs(f'{data_path}/masks_2/body', exist_ok=True)
    os.makedirs(f'{data_path}/masks_4/hair', exist_ok=True)
    os.makedirs(f'{data_path}/masks_4/body', exist_ok=True)

    for img_name in tqdm(img_names):
        basename = img_name.split('.')[0]
        img = np.asarray(Image.open(f'{data_path}/images/{img_name}'))
        h_old, w_old = img.shape[:2]
        mask_hair = np.asarray(Image.open(f'{data_path}/masks/hair/{basename}.png'))
        mask_body = np.asarray(Image.open(f'{data_path}/masks/body/{basename}.png'))
        if os.path.exists(f'{data_path}/masks/face/{basename}.png'):
            mask_face = np.asarray(Image.open(f'{data_path}/masks/face/{basename}.png'))
            # Check intersection between face and hair
            if ((mask_hair > 127) * (mask_face > 127)).sum() > (mask_body > 127).sum() * 0.1:
                print(f'Skipping frame {img_name}')
                continue

        Image.fromarray(img).resize((w_old // 2, h_old // 2), Image.BICUBIC).save(f'{data_path}/images_2/{img_name}')
        Image.fromarray(img).resize((w_old // 4, h_old // 4), Image.BICUBIC).save(f'{data_path}/images_4/{img_name}')
        Image.fromarray(mask_hair).resize((w_old // 2, h_old // 2), Image.BICUBIC).save(f'{data_path}/masks_2/hair/{img_name}')
        Image.fromarray(mask_body).resize((w_old // 2, h_old // 2), Image.BICUBIC).save(f'{data_path}/masks_2/body/{img_name}')
        Image.fromarray(mask_hair).resize((w_old // 4, h_old // 4), Image.BICUBIC).save(f'{data_path}/masks_4/hair/{img_name}')
        Image.fromarray(mask_body).resize((w_old // 4, h_old // 4), Image.BICUBIC).save(f'{data_path}/masks_4/body/{img_name}')


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)

    args = parser.parse_args()

    main(args)
