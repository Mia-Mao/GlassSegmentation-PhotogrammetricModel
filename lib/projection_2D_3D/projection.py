import os, sys, cv2
import numpy as np
import xml.dom.minidom as DM
import pickle as pkl
import scipy
import skimage
import skimage.transform

from PIL import Image
import time
import random
import multiprocessing
import copy
from shapely.geometry import Polygon
import shapely

def get_pixelPoint(vertice, R, center, K, distortion):

    [k1, k2, k3, p1, p2] = distortion
    vertice = vertice[1:] if len(vertice)==4 else vertice
    vertice = np.array(vertice).reshape((3,1))
    center = center.reshape((len(center),1))

    T = -np.dot(R, center)
    #camera coordination
    XYZ = np.dot(R, vertice) + T  
    u_ = XYZ[0,0]/XYZ[2,0]
    v_ = XYZ[1,0]/XYZ[2,0]

    r_2 = u_* u_ + v_ * v_
    u = u_*(1+k1*r_2+k2*r_2**2+k3*r_2**3)+2*p1*u_*v_+p2*(r_2+2*u_*u_) #矫正后的
    v = v_*(1+k1*r_2+k2*r_2**2+k3*r_2**3)+2*p2*u_*v_+p1*(r_2+2*v_*v_)

    px = K[0,0]*u + K[0,2]
    py = K[1,1]*v + K[1,2]

    return px, py

def write_into_obj(filter_projection_results, obj_save_dir, triangles):

    triangles_index_list = list(filter_projection_results.keys())

    all_vertice = []
    
    for triangles_index in triangles_index_list:
    
        triangles_3D = triangles[triangles_index]

        for vertice in triangles_3D:
                vertice_str = 'v '+str(vertice[0]) + ' ' + str(vertice[1]) + ' ' + str(vertice[2]) + '\n'
                all_vertice.append(vertice_str)

    all_vertice_set = list(set(all_vertice))
    all_vertice_set.sort(key = all_vertice.index)

    with open(obj_save_dir,'w') as f:

        f.writelines(all_vertice_set)

        vertice_indexes = [all_vertice_set.index(vertice_str) + 1 for vertice_str in all_vertice] #### 顶点编号从1开始
        vertice_indexes = np.array(vertice_indexes).reshape(-1,3)
        vertice_indexes = ['f ' + str(vi[0]) + ' ' + str(vi[1]) + ' ' + str(vi[2]) + '\n' for vi in vertice_indexes]
        f.writelines(vertice_indexes)
    
    f.close()

def get_normal(triangle):
    v11 = triangle[1, :] - triangle[0, :]
    v22 = triangle[2, :] - triangle[0, :]
    normal = np.cross(v11, v22)
    normal_unit = normal / np.linalg.norm(normal)
    return normal_unit, normal

def L2_dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a - b) ** 2))


def is_fg_in_image(tria_2D, segment_file_path):

    triangle_flag = 0

    try:
        image = Image.open(segment_file_path)
        segment_mat = np.array(image)

        flag_1_num = 0

        for [xx, yy] in tria_2D:
            flag = segment_mat[yy, xx]  

            if flag == 255:  #255:glass; 0:background
                flag_1_num += 1

        if flag_1_num >= 2: 
            triangle_flag = 1  #glass
        else:
            triangle_flag = -1  #background

    except:
        triangle_flag = -1

    return triangle_flag


def project_to_subregion(triangle, oblique_file, args, segments_path, K, distortion, height, width, stride, region_size):

    triangle_flag = 0 
    tria_2D = []
    segment_file_keyname = ' '
    
    segment_file = 'wh-building1_' + oblique_file
    oblimage_name = oblique_file 

    # rotation and camera center
    R = args[segment_file]['Rotation'] 
    center = args[segment_file]['Center']

    normal_unit, normal = get_normal(triangle)  # normal of triangle
    direct = center - triangle[0, :] #light
    direct_unit = direct / np.linalg.norm(direct)
    angle_cos = np.dot(direct_unit, normal_unit)
   
    if angle_cos < 0:  # angle > 90
        print('the angle between light and normal vector is larger than 90 !')


    else:
        # transfer to image coordinate, all the three vertices should locate in the image, otherwise, go into the next image
        delta_x = 1 
        delta_y = 0

        col_list = []
        row_list = []
        
        for vertice in triangle:
            # vertice = np.array([975.7185, 726.5926, 86.0663])       
            px, py = get_pixelPoint(vertice, R, center, K, distortion) #px, py cor in oblique image
            px = int(round(px+delta_x))
            py = int(round(py+delta_y))

            if px < width and px >= 0 and py <height and py >= 0:
                # print('Coord of px and py in oblique image:')                
                col_num = int(px/stride)
                row_num = int(py/stride)

                xx = int(px - stride * col_num)
                yy = int(py - stride * row_num) 

                col_list.append(col_num)
                row_list.append(row_num)        

                tria_2D.append([xx, yy])   
            
            else:
                print('Projected point is not in the oblique image.')
                break

        # exit(0)
        if len(col_list)==3 and len(row_list)==3:  #判断col_list，row_list是否有3个点 
        
            if col_list[0] == col_list[1] == col_list[2] and row_list[0] == row_list[1] == row_list[2]:
                # print('col_num == row_num, processing projected point....')
                segment_file_name = oblimage_name + '_' +str(col_list[0]) + '_' +str(row_list[0]) + '.png'  
                segment_file_path = os.path.join(segments_path, oblimage_name, segment_file_name) 

                segment_file_keyname = 'wh-building1_' + segment_file_name  

                if os.path.exists(segment_file_path): 
                    triangle_flag = is_fg_in_image(tria_2D, segment_file_path) 

            else:
                print('col_num != row_num.')
        else:
            print('col_list is empty!')

    return triangle_flag, tria_2D, segment_file_keyname


def tri_filter_by_distance(projection_results, triangles, para):

    projection_results_copy = copy.deepcopy(projection_results)

    key_list = list(projection_results.keys()) #triangle 3D index list
    # print('key_list:', key_list)
    triangles_3D_number = int(len(key_list))

    args = para['args'] #AT info

    for index_i in range(triangles_3D_number-1): # triangle_3D index
        
        triangles_3D_i = projection_results[key_list[index_i]] 
                index_k = index_i + 1
        triangles_3D_j = projection_results[key_list[index_k]]

        oblimage_subregion_listi = triangles_3D_i.keys()
        oblimage_subregion_listj = triangles_3D_j.keys()

        same_oblimage_subregion = [x for x in oblimage_subregion_listi if x in oblimage_subregion_listj] #两个列表表都存在相同的subregion

        if len(same_oblimage_subregion):
            for oblimage_subregion in same_oblimage_subregion:  
            # iou
                triangle_2D_i = triangles_3D_i[oblimage_subregion]
                triangle_2D_j = triangles_3D_j[oblimage_subregion]
                
                iou = compute_tri_iou(triangle_2D_i, triangle_2D_j)

                if iou >= 0.5: 

                    # parameters:
                    oblimage_name = oblimage_subregion.split('_')[1]  
                    segment_file = 'wh-building1_' + oblimage_name 
                    center = args[segment_file]['Center']

                    triangles_3D_A = triangles[key_list[index_i]] 
                    triangles_3D_B = triangles[key_list[index_k]]

                    center_A = np.mean(triangles_3D_A, 0) 
                    center_B = np.mean(triangles_3D_B, 0)

                    distance_A = L2_dist(np.mean(center_A, axis=0), center)
                    distance_B = L2_dist(np.mean(center_B, axis=0), center)

                    if distance_A > distance_B:
                        index_A = key_list[index_i] 

                        if oblimage_subregion in projection_results_copy[int(index_A)].keys():
                            del projection_results_copy[int(index_A)][oblimage_subregion]

                    else:
                        index_B = key_list[index_k]
                        if oblimage_subregion in projection_results_copy[int(index_B)].keys():
                            del projection_results_copy[int(index_B)][oblimage_subregion]

    return projection_results_copy
  

def compute_tri_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(3, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(3, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            # union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 1
            
            iou1 = float(inter_area) / float(union_area)

            if min(poly1.area, poly2.area) > 0:
                iou2 = float(inter_area) / float(min(poly1.area, poly2.area))
            else:
                iou2 = 1
            
            iou = max(iou1, iou2)
    
        except shapely.geos.TopologicalError:
            iou = 0
    
    return iou 


def projection(triangles, original_oblimg_list, para, segments_path, region_size=512, step=0.8):
    
    args = para['args'] 

    f = para['f']
    K = para['K']

    Coord = para['Coord']
    distortion = para['Distortion']
    width = para['Image_width']
    height = para['Image_height']
    
    stride = int(region_size*step)
    project_result = {}

    for triangle_3D_index, triangle_3D in enumerate(triangles): #3D triangles

        point_coord = {}
        
        triangle_flag_glass_num = 0
        triangle_flag_backg_num = 0

        for oblique_file in original_oblimg_list: #大图 oblique image 无后缀

            # oblique_file = 'DSC00166'

            ##project to subregion
            triangle_flag, tria_2D, segment_file_keyname = project_to_subregion(triangle_3D, oblique_file, args, segments_path,  K, distortion, height, width, stride, region_size)

            if triangle_flag == 1: 

                triangle_flag_glass_num += 1
                point_coord[segment_file_keyname] = tria_2D 
                # print('**************************************************************************')

            elif triangle_flag == -1: 
                triangle_flag_backg_num += 1

        if triangle_flag_glass_num + triangle_flag_backg_num > 0:
            
            glass_ratio = float(triangle_flag_glass_num / (triangle_flag_glass_num + triangle_flag_backg_num))
        
            if glass_ratio > 0.1:  #triangle_flag_backg_num

                project_result[triangle_3D_index] = point_coord
            # print('*****the triangle belongs to the glass area.**********')

    return project_result 