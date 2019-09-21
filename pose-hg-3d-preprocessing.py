#pose-hg-3d-preprocessing.py
#author: Shih-Yao (Mike) Lin
#email: shihyaolin.tencent.com
#date: 2019-09-19
#input: image folder path & kjson folder path (all in .h5 format)
#output: cropped images with zero padding boundary (all the person in the image will be located at the center of each image)
#usage: python3 pose-hg-3d-preprocessing.py -i [image path] -j [kjson path] -o [output image path]
#usage: python3 pose-hg-3d-preprocessing.py -i /data/yushengxie/docker_scoliosis_gait/jpgs/ -j /data/yushengxie/docker_scoliosis_gait/reid_kjsons/ -o /data/mikelin/output/docker_scoliosis_gait/c_imgs
#usage: nohup python3 pose-hg-3d-preprocessing.py -i /data/yushengxie/docker_scoliosis_gait/jpgs/ -j /data/yushengxie/docker_scoliosis_gait/reid_kjsons/ -o /data/mikelin/output/docker_scoliosis_gait/c_imgs > pre_processing.log 2>&1 &

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

import statistics

POSDIM = 21
FACEDIM = 70
HANDDIM = 21
fix_size = 500

def reset(reset_path):
    path = reset_path
    if os.path.isdir(path):
        shutil.rmtree(path)
        makedirs(path)
    else:
        makedirs(path)

def find_bbox_and_center(keypoints, img_w, img_h):

	x = []
	y = []
	c = 0
	padding_w = 150

	bbox = []
	center = []
	for  i, key in enumerate(keypoints):
		if i > 24:
			break
		if key[0] > 0 and key[1] > 0:
			x.append(key[0])
			y.append(key[1])

	if x != [] and y != [] and len(x)>16 and len(y)>16 :
		c_x = int(statistics.median(x))
		c_y = int(statistics.median(y))
		center = (c_x, c_y)

		b_x_0 = int(min(x))-padding_w
		b_x_1 = int(max(x))+padding_w
		b_y_0 = int(min(y))-padding_w
		b_y_1 = int(max(y))+padding_w

		if b_x_0 < 0:
			b_x_0 = 0
		if b_x_1 >img_w:
			b_x_1 = img_w
		if b_y_0 < 0:
			b_y_0 = 0
		if b_y_1 > img_h:
			b_y_1 = img_h


		bbox = [(b_x_0, b_y_0), (b_x_1, b_y_1)]


	return center, bbox

# def bbox(keypoints, )


def preprocessing(input_img_folder, input_keypoint_folder, output_folder):

	# parameters setting
	radius = 10
	color = (0, 0, 255)
	thickness = -1

	reset(output_folder)

	img_h5_list = sorted(glob.glob(input_img_folder+"/2019090*.h5"))
	print("image files: "+ str(len(img_h5_list)))

	keypoint_h5_list = sorted(glob.glob(input_keypoint_folder+"/2019090*.h5"))
	print("kjson files: " +str(len(keypoint_h5_list)))


	for i, fi in enumerate(img_h5_list):

		# 1. reset all the output folder
		file_name = fi[len(input_img_folder):]
		out = output_folder+"/"+file_name
		print("process "+file_name)
		reset(out)
		reset(output_folder+"/_bbox_"+file_name)

		# 2. load data (keypoints&imgs)
		## 2,1 load images
		f = h5py.File(img_h5_list[i], 'r')
		imgs = f['binary_jpg']
		img_list = []
		for j, img in enumerate(imgs):
			pil_img = Image.open(io.BytesIO(img)).convert('RGB') 
			open_cv_image = np.array(pil_img) 
			# Convert RGB to BGR 
			open_cv_image = open_cv_image[:, :, ::-1].copy() 
			img_list.append(open_cv_image)
		# 2.2 load keypoint
		f = h5py.File(keypoint_h5_list[i], 'r')
		keypoints = f['keypoint']
		keypoint_list = []
		for j, keypoint in enumerate(keypoints):
			keypoint_list.append(keypoint)

		# 3. process each frame
		frame_count = 0
		for j, img in enumerate(img_list):

			keypoint = keypoint_list[j]
			img_w = len(img[0])
			img_h = len(img)

			# 3.0 calculate bbox and bbox center
			center, bbox = joint_bbox_and_center(keypoint, img_w, img_h)

			if bbox != []:
				print(j)

				## 3.0.0 plot bbox
				# img_copy = img.copy()
				# cv2.circle(img_copy, center, radius, color, thickness) 
				# cv2.rectangle(img_copy, bbox[0], bbox[1], (0,255,0), 2)
				# cv2.imwrite(output_folder+"/_bbox_"+file_name+"/%04d.jpg"%j,img_copy)

				# 3.1 resize image
				resized_img = img.copy()
				resized_img = img[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0] ]
				w = len(resized_img[0])
				h = len(resized_img)
				new_w = int(fix_size * w /h)
				new_h = fix_size

				if new_w % 2 == 1:
					new_w += 1
				resized_img = cv2.resize(resized_img, (new_w, new_h), interpolation = cv2.INTER_AREA)
				# cv2.imwrite(out+"/__%04d.jpg"%j,resized_img)

				# 3.2 zero padding
				blank_image = np.zeros(shape=[fix_size, fix_size, 3], dtype=np.uint8)
				blank_center = int(fix_size/2)
				x_s = int(blank_center-new_w/2)
				x_e = int(blank_center+new_w/2)
				y_s = 0
				y_e = fix_size

				new_bbox = (x_s, x_e, y_s, y_e)
				blank_image[y_s:y_e,x_s:x_e] = resized_img[0:new_h, 0:new_w]
				cv2.imwrite(out+"/%04d.jpg"%j,blank_image)
				frame_count +=1

			print(file_name+" has been processed. Total frames:"+str(frame_count))


if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_img_h5_folder", type=str)
	parser.add_argument("-j", "--input_keypoint_h5_folder", type=str)
	parser.add_argument("-o", "--output_folder", type=str)

	args = parser.parse_args()

	preprocessing(args.input_img_h5_folder, args.input_keypoint_h5_folder, args.output_folder)