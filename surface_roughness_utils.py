
import PyIPSDK
import PyIPSDK.IPSDKIPLAdvancedMorphology as advmorpho
import PyIPSDK.IPSDKIPLShapeAnalysis as shapeanalysis
import PyIPSDK.IPSDKIPLArithmetic as arithm
import PyIPSDK.IPSDKIPLBinarization as bin
import PyIPSDK.IPSDKIPLBinarization as PyIPSDKbin
import PyIPSDK.IPSDKIPLLogical as logic
import PyIPSDK.IPSDKIPLMorphology as morpho
import PyIPSDK.IPSDKIPLUtility as util
import PyIPSDK.IPSDKIPLFiltering as filtering
import PyIPSDK.IPSDKIPLIntensityTransform as itrans
import PyIPSDK.IPSDKIPLGeometricTransform as gtrans
import PyIPSDK.IPSDKIPLGlobalMeasure as glbmsr

from ipsdk import imageUtils

import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from datetime import datetime
from functools import partial

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import colors

from sklearn.linear_model import LinearRegression

import sys

sys.path.append('/mnt/XNAS/data/USERS/helmerking/code/99_in_development/Surfaceroguhness/python_scripts/00_code/')
from create_distance_surface import create_dist_from_mask

#sys.path.append('/mnt/XNAS/data/USERS/helmerking/code/99_in_development/Surfaceroguhness/python_scripts/98_past_usecases/')
#from FB_distance_surface_calculation import create_rendering_surfaces_from_distsurface_and_mask_ADDMASK

sys.path.append('/mnt/XNAS/data/USERS/helmerking/code/99_in_development/Surfaceroguhness/python_scripts/99_archive/')
from analyze_dist_surface import calc_Ra

sys.path.append('/mnt/XNAS/data/USERS/mattern/04_Coding/01_utils/')
from utils import *


#sys.path.append('/mnt/XNAS/data/USERS/helmerking/code/99_in_development/Surfaceroguhness/python_scripts/98_past_usecases/')
#from FB_distance_surface_calculation import create_rendering_surfaces_from_distsurface_and_mask_ADDMASK


#creates the distance of the surface to the mask. Giving the surface_deviation as float32 image
#credit to PH
def create_dist_from_mask(mask, surface, MAX_DIST=999, distance_map_type='int'):
    '''

    output = create_dist_from_mask(mask, surface, MAX_DIST=9999999)
    _______________________________________________________________________________________________
    Input:
    - mask	; type PyIPSDK.BinaryImage	; mask of the reference volume to calculate the distance of the surface to
    - surface	; type PyIPSDK.BinaryImage	; mask of the surface of which  to calculate the distance to the reference volume
    - MAX_DIST	; type int			; maximal distance to calculate (can be used to save computation time, if not all values are of interest)
    - distance_map_type ; type str ; 'int' or 'float' decides which kind of distance map to calculate
    ______________________________________________________________________________________________
    Output:
    - dist = PyIPSDK.float32 image containing the distance to the reference volume on every point of the surface
    _______________________________________________________________________________________________
    
    '''
    mask_i =logic.logicalNotImg(mask)
    mask =logic.logicalNotImg(mask_i)
    if distance_map_type == 'float':
        dm3d = PyIPSDK.createImage(mask, PyIPSDK.eIBT_Real32)
        morpho.distanceMap3dImg(mask, float(MAX_DIST) , dm3d)
        dm3d_i = PyIPSDK.createImage(mask_i, PyIPSDK.eIBT_Real32)
        morpho.distanceMap3dImg(mask_i, float(MAX_DIST) , dm3d_i)
    elif distance_map_type == 'int':
        dm3d = morpho.distanceMap3dImg(mask, float(MAX_DIST))
        dm3d_i = morpho.distanceMap3dImg(mask_i, float(MAX_DIST))
	    
    dm3d = arithm.subtractScalarImg(1.0,dm3d)	#this gets rid of the -1 to 1 jump at the surface between the two distance maps
    dm3d_mask = bin.darkThresholdImg(dm3d, 0.0)
    dm3d = logic.maskImg(dm3d,dm3d_mask)
    dist_neg = logic.maskImg(dm3d,surface)
    dist_pos = logic.maskImg(dm3d_i, surface)
    dist = arithm.addImgImg(dist_pos, dist_neg)

    return dist

#Calculation of Ra in x y or z slices! It might be quicker to use the dedicated stats mask msr 2d function for speed.
# credit to PH
def calc_Ra_slicewise_in_axis(dist_surface, surface_mask, direction='z'):

    
	#vol_shape = dist_surface.array.shape
	#Ra_values = []
	slice_mean = []
	slice_stdDev = []
	abs_dist_surface = arithm.absImg(dist_surface)
	if direction == 'z':
		axis_length = dist_surface.array.shape[0]
	elif direction == 'y':
		axis_length = dist_surface.array.shape[1]
	elif direction == 'x':
		axis_length = dist_surface.array.shape[2]
	else:
		print('Unexpected axis argument for calculating slicewise Ra!!!')
	for i in range(axis_length):
		if direction == 'z':
			dist_surface_slice = util.getROI3dImg(abs_dist_surface, 0, 0, i, abs_dist_surface.array.shape[2], abs_dist_surface.array.shape[1], 1)
			surface_mask_slice = util.getROI3dImg(surface_mask, 0, 0, i, surface_mask.array.shape[2], surface_mask.array.shape[1], 1)
		elif direction == 'y':
			dist_surface_slice = util.getROI3dImg(abs_dist_surface, 0, i, 0, abs_dist_surface.array.shape[2], 1, abs_dist_surface.array.shape[0])
			surface_mask_slice = util.getROI3dImg(surface_mask, 0, i, 0, surface_mask.array.shape[2], 1, surface_mask.array.shape[0])
		elif direction == 'x':
			dist_surface_slice = util.getROI3dImg(abs_dist_surface, i, 0, 0, 1,abs_dist_surface.array.shape[1], abs_dist_surface.array.shape[0])
			surface_mask_slice = util.getROI3dImg(surface_mask, i, 0, 0, 1,surface_mask.array.shape[1], surface_mask.array.shape[0])
		stats = glbmsr.statsMaskMsr3d(dist_surface_slice,surface_mask_slice)
		mean_val = stats.mean
		stdDev_val = stats.stdDev
		slice_mean.append(mean_val)
		slice_stdDev.append(stdDev_val)
	return slice_mean, slice_stdDev


#calculate Ra of a 3d dist_suface
#credit to PH
def calc_Ra(dist_surface, surface_mask):
	abs_dist_surface = arithm.absImg(dist_surface)
	stats = glbmsr.statsMaskMsr3d(abs_dist_surface,surface_mask)
	mean_val = stats.mean
	#stdDev_val = stats.stdDev
	return(mean_val)

# only here because of the import error
#creates a volume that can be rendered well by avizo and adds the mask so that the structure of the volume of the distancemap can be rendered aswell
def create_rendering_surfaces_from_distsurface_and_mask_ADDMASK(dist_surface, mask, se_size=3):
	mask = PyIPSDK.fromArray(mask)
	mask = bin.lightThresholdImg(mask, 1.0)
	dist_surface = PyIPSDK.fromArray(dist_surface)
	dist_surface = arithm.absImg(dist_surface)
	dist_surface = util.convertImg(dist_surface, PyIPSDK.eImageBufferType.eIBT_UInt8)
	structuringElement= PyIPSDK.sphericalSEXYZInfo(se_size)
	dist_surface = morpho.dilate3dImg(dist_surface, structuringElement)
	dist_surface = logic.maskImg(dist_surface,mask)
	dist_surface = arithm.addImgImg(dist_surface, mask)
	dist_surface = util.convertImg(dist_surface, PyIPSDK.eImageBufferType.eIBT_UInt8)
	return dist_surface

####################################################################################################################
# plane creation


def fit_plane_within_bounding_box(image):
    # Step 1: Find indices where the array has value 1
    points = np.argwhere(image == 1)
        
    # If there are no points, return an empty array or handle accordingly
    if len(points) == 0:
        return np.zeros_like(image)
    
    # Step 2: Separate into x, y, and z coordinates
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    
    # Step 3: Find the bounding box for the points
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()
        
    # Step 4: Perform linear regression to fit Z = aX + bY + d
    XY = np.column_stack((X, Y))
    model = LinearRegression().fit(XY, Z)
    
    # Coefficients for the plane equation Z = aX + bY + d
    a, b = model.coef_
    d = model.intercept_
        
    # Step 5: Create a new image for the fitted plane
    plane_image = np.zeros_like(image)
    
    # Step 6: Populate the bounding box region in the new image with the fitted plane
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            # Calculate the Z value of the plane at (x, y)
            z_plane = int(round(a * x + b * y + d))
            
            # Only assign if z_plane is within the bounding box in Z dimension
            if z_min <= z_plane <= z_max:
                plane_image[x, y, z_plane] = 1  # Mark as part of the fitted plane
                

    return plane_image


def dilate_object_one_direction(mask_array, direction='x', nb_pixels=1):
    
    if direction == 'x':
        dim=2
    elif direction == 'y':
        dim=1
    else:
        dim=0
        
    # Find points where the mask is 1
    points = np.argwhere(mask_array == 1)
    
    # Dilate each point in the specified direction by `nb_pixels`
    for offset in range(1, nb_pixels + 1):
        # Create a copy of the points array and add the offset
        new_points = points.copy()
        new_points[:, dim] += offset
        
        # Filter out points that go out of bounds
        within_bounds = (
            (new_points[:, 0] >= 0) & (new_points[:, 0] < mask_array.shape[0]) &
            (new_points[:, 1] >= 0) & (new_points[:, 1] < mask_array.shape[1])
        )
        new_points = new_points[within_bounds]
        
        # Update the mask array
        mask_array[tuple(new_points.T)] = 1

    return mask_array

####################################################################################################################
# filtering functions

def filter_thin_shapes(mask, small_structure_size=1, min_nb_pixels=10000):

	structuringElement= PyIPSDK.sphericalSEXYZInfo(small_structure_size)

	big_structures_with_artifacts = morpho.erode3dImg(mask, structuringElement)

	
	#big_structures_with_artifacts_cc = advmorpho.connectedComponent3dImg(big_structures_with_artifacts)
	#big_structures_cc = advmorpho.keepBigShape3dImg(big_structures_with_artifacts_cc, n_big_shapes)
	
	#big_structures = bin.lightThresholdImg(big_structures_cc, 1)

	big_structures = filter_components_volume(big_structures_with_artifacts, min_nb_pixels=min_nb_pixels)


	#dilate before filtering
	structuringElement= PyIPSDK.sphericalSEXYZInfo(small_structure_size*2)
	big_structures = morpho.dilate3dImg(big_structures, structuringElement)

	big_structures = logic.bitwiseAndImgImg(mask, big_structures)

	return big_structures

def filter_components_volume(mask, min_nb_pixels=0, max_nb_pixels=np.inf, connexity=26):

	
	if connexity == 6:
		ipsdk_connexity = PyIPSDK.eNeighborhood3dType.eN3T_6Connexity
	else:
		ipsdk_connexity = PyIPSDK.eNeighborhood3dType.eN3T_26Connexity
     		
	mask_cc = advmorpho.connectedComponent3dImg(mask, ipsdk_connexity)
	mask_filtered = shapeanalysis.shapeFiltering3dImg(mask_cc, mask_cc, f"NbPixels3dMsr > {min_nb_pixels}")

 
	if max_nb_pixels < np.inf:
		mask_filtered = shapeanalysis.shapeFiltering3dImg(mask_filtered, mask_filtered, f"NbPixels3dMsr < {max_nb_pixels}")
  
	mask_filtered = bin.lightThresholdImg(mask_filtered, 1.0)

	return mask_filtered

def filter_components_area2D(mask, min_nb_pixels=9000):
    
	mask_cc = advmorpho.connectedComponent2dImg(mask)
	mask_filtered = shapeanalysis.shapeFiltering2dImg(mask_cc, mask_cc, f"NbPixels2dMsr > {min_nb_pixels}")

	mask_filtered = bin.lightThresholdImg(mask_filtered, 1.0)

	return mask_filtered


def filter_cylinder(mask, margin):
	
	xsz = mask.getSizeX()
	ysz = mask.getSizeY()
	zsz = mask.getSizeZ()
	
	# Define the dimensions of the 3D grid
	z_dim = zsz  # Height of the cylinder
	xy_dim = xsz  # Width and height of the square base grid
	radius = xsz//2 - margin  # Radius of the cylinder

	# Create a 3D numpy array filled with zeros
	cylinder_array = np.zeros((z_dim, xy_dim, xy_dim), dtype=bool)

	# Define the center of the cylinder's base
	center_x, center_y = xy_dim // 2, xy_dim // 2

	# Create a grid for the x and y coordinates
	x, y = np.meshgrid(np.arange(xy_dim), np.arange(xy_dim), indexing='ij')

	# Calculate the distance from the center for each point
	distance_from_center = (x - center_x)**2 + (y - center_y)**2

	# Set points within the radius to 1
	cylinder_array[:, distance_from_center <= radius**2] = 1

	cylinder_mask = PyIPSDK.fromArray(cylinder_array)
 
	cylinder_mask = bin.thresholdImg(cylinder_mask, 1, 1)
	
	# mask_filtered = logic.bitwiseAndImgImg(mask, cylinder_mask)
 
	mask_filtered = arithm.multiplyImgImg(mask, cylinder_mask)

	return mask_filtered, cylinder_mask




####################################################################################################################
# surface functions


def compute_surface_regions(mask, direction='x1', filter_volume=True, filter_lamella_regions=True, 
							  min_nb_pixels_sobel=10000, max_nb_pixels_sobel=30000, 
							    min_nb_pixels_lamellas=0, max_nb_pixels_lamellas=np.inf):

	sobel = filtering.sobelGradient3dImg(mask)

	sobelx1 = bin.lightThresholdImg(sobel[0], 10)
	sobelx2 = bin.darkThresholdImg(sobel[0], -10)
	
	sobelx = logic.bitwiseOrImgImg(sobelx1, sobelx2)	

	sobely1 = bin.lightThresholdImg(sobel[1], 10)
	sobely2 = bin.darkThresholdImg(sobel[1], -10)

	sobely = logic.bitwiseOrImgImg(sobely1, sobely2)
 
	sobelz1 = bin.lightThresholdImg(sobel[2], 10)
	sobelz2 = bin.darkThresholdImg(sobel[2], -10)

	sobelz = logic.bitwiseOrImgImg(sobelz1, sobelz2)


	if direction == 'x1':
		sobel = sobelx1
	elif direction == 'x2':
		sobel = sobelx2
	elif direction == 'y1':
		sobel = sobely1
	elif direction == 'y2':
		sobel = sobely2
	elif direction == 'z1':
		sobel = sobelz1
	elif direction == 'z2':
		sobel = sobelz2
	elif direction == 'x':
		sobel = sobelx
	elif direction == 'y':
		sobel = sobely
	elif direction == 'z':
		sobel = sobelz
  
	xsz = mask.getSizeX()
	ysz = mask.getSizeY()
	zsz = mask.getSizeZ()

	cylinder = np.zeros((zsz, xsz, ysz), dtype=bool)
	cylinder[:, ysz//2, xsz//2] = 1

	ones = np.ones((zsz, xsz, ysz), dtype=bool)
	ones = PyIPSDK.fromArray(ones)

	cylinder = PyIPSDK.fromArray(cylinder)

	cylinder_mask, _ = advmorpho.seededDistanceMap3dImg(ones, cylinder)
	cylinder_mask = bin.thresholdImg(cylinder_mask, 0, 150)

	sobel_filtered = logic.bitwiseAndImgImg(sobel, cylinder_mask)

	if filter_volume:
		sobel_filtered = filter_components_volume(sobel_filtered, min_nb_pixels=min_nb_pixels_sobel, max_nb_pixels=max_nb_pixels_sobel)

	sobel_filtered = morpho.dilate3dImg(sobel_filtered, PyIPSDK.sphericalSEXYZInfo(1))


	if filter_lamella_regions:
		lamellas = filter_for_lamellas(mask, small_structure_size=3, min_nb_pixels=min_nb_pixels_lamellas, max_nb_pixels=max_nb_pixels_lamellas)
		lamellas_ROI = morpho.dilate3dImg(lamellas, PyIPSDK.sphericalSEXYZInfo(1))	
		surface_ROI = logic.bitwiseAndImgImg(sobel_filtered, lamellas_ROI)
		surface_ROI = advmorpho.connectedComponent3dImg(surface_ROI)
	else:
		surface_ROI = advmorpho.connectedComponent3dImg(sobel_filtered)
    #add other components later
	return surface_ROI





