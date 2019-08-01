#pose-hg-3d-preprocessing.py
#author: Shih-Yao (Mike) Lin
#email: shihyaolin.tencent.com
#date: 2019-07-31
#usage python pose-hg-3d-preprocessing.py -i /Users/shihyaolin/Documents/data/scoliosis/jpgs/ -j /Users/shihyaolin/Documents/data/scoliosis/reid_kjsons/ -o /Users/shihyaolin/Documents/data/scoliosis/cropped/20190730_0_5.walk/ 

import glob
import argparse
import numpy as np
import os, sys
from os import listdir, makedirs
import shutil
import cv2

import h5py
import io
from PIL import Image

import random

random.seed( 30 )

POSDIM = 21
FACEDIM = 70
HANDDIM = 21

def reset(reset_path):
    path = reset_path
    if os.path.isdir(path):
        shutil.rmtree(path)
        print("remove existing "+path)
        makedirs(path)
        print("create folder: "+path)
    else:
        makedirs(path)
        print("create foder: "+path)


def preprocessing(input_img_folder, input_keypoint_folder, output_folder):

	reset(output_folder)

	img_h5_list = sorted(glob.glob(input_img_folder+"/*.h5"))
	print(len(img_h5_list))

	keypoint_h5_list = sorted(glob.glob(input_keypoint_folder+"/*.h5"))
	print(len(keypoint_h5_list))

	for i, fi in enumerate(img_h5_list):

		file_name = fi[len(input_img_folder):]
		out = output_folder+"/"+file_name
		reset(out)

		# load data
		f = h5py.File(img_h5_list[i], 'r')
		imgs = f['binary_jpg']

		img_list = []
		for j, img in enumerate(imgs):
		    pil_img = Image.open(io.BytesIO(img)).convert('RGB') 
		    open_cv_image = np.array(pil_img) 
		    # Convert RGB to BGR 
		    open_cv_image = open_cv_image[:, :, ::-1].copy() 
		    img_list.append(open_cv_image)

	    # load keypoint
		f = h5py.File(keypoint_h5_list[i], 'r')
		keypoints = f['keypoint']

		keypoint_list = []

		for j, keypoint in enumerate(keypoints):
		    keypoint_list.append(keypoint)

		
		width = int(abs(keypoint_list[100][8][1]-keypoint_list[100][16][1]))+40

		print(width)
		while width <40 or width> 200:
			idx = random.randint(50,200)
			width = int(abs(keypoint_list[idx][8][1]-keypoint_list[idx][16][1]))+40
			print(width)


		cropped_img_list = []

		for j, img in enumerate(img_list):
		    center = (keypoint_list[j][8][0], keypoint_list[j][8][1])
		#     cv2.circle(img, center, 10, (0,255,0), -1)

		    x_0 = int(center[0]-width)
		    y_0 = int(center[1]-width)
		    x_1 = int(center[0]+width)
		    y_1 = int(center[1]+width)
		    
		    if x_0 >0 and x_1 < len(img[1]):
		        img2 = img[y_0:y_1, x_0:x_1]
		        cv2.imwrite(out+"/%04d.jpg"%j,img2)


if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_img_h5_folder", type=str)
	parser.add_argument("-j", "--input_keypoint_h5_folder", type=str)
	parser.add_argument("-o", "--output_folder", type=str)

	args = parser.parse_args()

	preprocessing(args.input_img_h5_folder, args.input_keypoint_h5_folder, args.output_folder)