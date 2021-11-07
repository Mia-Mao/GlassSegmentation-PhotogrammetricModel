import _init_paths
import os
import numpy as np
import matplotlib.pyplot as plt
import colorsys,random
from read_data.OBJ_PARSER import *
from read_data.XML_PARSER import *
from read_data.MASK_PARSER import *
from projection_2D_3D.projection import *
import pickle

def generate_colors(N = 10):
	HSV_tuples = [(x*1.0/N, 0.7, 0.7) for x in range(N)]
	RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
	random.shuffle(RGB_tuples)
	return RGB_tuples

def get_oblimg_list(file_folder):

	original_oblimg_list = os.listdir(file_folder)

	return original_oblimg_list

def main(AT_file, model_list, model_path, segments_path, pkl_save_folder, obj_save_folder):
	if not os.path.exists(model_path):
		raise(model_path + " is not exists, please check.")

	original_oblimg_list = get_oblimg_list(segments_path)
		
	####Parsing AT fiel to obtain the parameters 解析内外参，获得所有倾斜影像的空三信息
	para = parse_xml(AT_file)
	args = para['args'] #AT info


	for model in model_list:
		print('\n ........................current model is: ' + model + '.............................')

		model_name, model_ext = os.path.splitext(model)

	####Parsing 3D model(.obj) to get coordinates of trangle mesh 读取三维坐标	
		model_file = os.path.join(model_path, model)
		obj_parser = OBJ_PARSER(model_file)
		triangles, faces, vertices = obj_parser.get_triangles()
		print('reading data done .....')
		# print('triangles:', triangles)

		#### stage 1: project 3D mesh to 2D image#############
		print('stage 1: project 2D segments to 3D model .........')
		projection_results = projection(triangles, original_oblimg_list, para, segments_path)
		print('Finish projection.....\n')


		pkl_temp_file = pkl_save_folder + model_name + '_temp.pkl'
		
	# 	# save the temp result 把results写进pkl，临时的result
		with open(pkl_temp_file, 'wb') as temp_pkl_f:
			pickle.dump(projection_results, temp_pkl_f)
		temp_pkl_f.close()

		# **************************************************************************
		# stage2: remove the wrong correspondence
		# 判断projection_results中是否有重叠的三角形，如果重叠三角形iou>0.5 同时，若有多个三角形对应同一像素，根据距离选出可视三角形

		print('len(projection_results):', len(projection_results))
		if len(projection_results)>=2:
			filter_projection_results = tri_filter_by_distance(projection_results, triangles, para)
		else:
			filter_projection_results = projection_results

		pkl_final_result = pkl_save_folder + model_name + '_final.pkl'
		with open(pkl_final_result, 'wb') as pkl_final:
			pickle.dump(filter_projection_results, pkl_final)
		pkl_final.close()

		obj_save_dir =  obj_save_folder + model_name + '_segmentResult.obj'


		#save the final glass segmentation result in .obj file. 
		write_into_obj(filter_projection_results, obj_save_dir, triangles)


if __name__ == "__main__":
	
	model_path = '3D Models/demo/'  # 3D model folder
	model_list = ['demo.obj'] # 3D model file name
	AT_file = '3D Models/demo/AT file/AT-demo.xml' # AT_file of 3D model
	segments_path = 'obliqueimage_detection_result/' # path for glass detection results in oblique image.

	# folder to save the results.
	pkl_save_folder = './pkl_result/demo/'
	obj_save_folder = './obj_result/demo/'

	if not os.path.exists(obj_save_folder):
		os.makedirs(obj_save_folder)

	if not os.path.exists(pkl_save_folder):
		os.makedirs(pkl_save_folder)

	main(AT_file, model_list, model_path, segments_path, pkl_save_folder, obj_save_folder) 
