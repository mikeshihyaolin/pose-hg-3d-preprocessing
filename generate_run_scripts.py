#generate_run_scripts
#author: Shih-Yao (Mike) Lin
#email: shihyaolin.tencent.com
#date: 2019-07-31
#usage python generate_run_scripts.py -s ~/Documents/data/scoliosis/cropped/ -t ~/Documents/data/scoliosis/3d_pose_res/

import glob
import argparse
import os, sys
from os import listdir, makedirs
import shutil

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


def generate_scripts(source_folder, target_folder):
	
	reset(target_folder)

	os.chdir(source_folder)
	all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
	print(all_subdirs)

	f= open("/Users/shihyaolin/Documents/mike-test/pose-hg-3d-preprocessing/run_scripts.sh","w")

	for folder in all_subdirs:
		cmd = "python demo.py --demo "+source_folder+folder+" --output_path "+target_folder+"/"+folder+"  --gpus -1  --load_model ../models/fusion_3d_var.pth "
		f.write(cmd+"\n")

	f.close()
	




# # python demo.py --demo /Users/shihyaolin/Documents/data/scoliosis/cropped/20190730_0_5.walk_jpg.h5  --gpus -1  
# --load_model ../models/fusion_3d_var.pth --output_path /Users/shihyaolin/Documents/data/scoliosis/3d_pose_res/


if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--source_folder", type=str)
	parser.add_argument("-t", "--target_folder", type=str)

	args = parser.parse_args()

	generate_scripts(args.source_folder, args.target_folder)