#%%
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

import ipywidgets as widgets
from IPython.display import display, clear_output

import sys


def get_numpy_values(val):
	return np.array(val.getMeasureResult().getColl(0))



############################################################################################################
#%% plotting functions


# function from JP imageUtils, adjusted for different figure size
def showSlice(img,sliceN=None,lims=None,dim=0,**kwargs):
	if sliceN is None:
		sliceN = img.array.shape[dim] // 2

	slices = [slice(None), slice(None), slice(None)]

	if img.getVolumeGeometryType() == PyIPSDK.eVGT_2d:
		slices.pop()
	else:
		slices[dim] = sliceN

	image = img.array
	if image.ndim > 2:
		image = np.squeeze(image)
	if image.ndim > 3:
		raise ValueError("image has too many non-singleton dimensions. Shape after squeezing: {}".format(image.shape))
	if image.ndim > 2:
		if sliceN is None:
			sliceN = image.shape[dim] // 2
		slices = [slice(None), slice(None), slice(None)]
		slices[dim] = sliceN
		image = image[tuple(slices)]
  

	fig = plt.figure(figsize=(8, 6))
	ax = plt.axes()

	if 'cmap' not in kwargs:
		kwargs['cmap'] = plt.cm.Greys_r

	if lims is None:
		im = ax.imshow(image, **kwargs)
	else:
		im = ax.imshow(image, vmin=lims[0], vmax=lims[1], **kwargs)
  	
	showToolTip = kwargs.get('showToolTip', False)
	if showToolTip:
		ax.format_coord = Formatter(im)
	plt.draw()
	fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # Remove all outer padding
	plt.show(block=False)
	return im

def showSliceBeforeAfter(mask1, mask2, sliceN=None,lims=None,dim=0,**kwargs):
    
	if sliceN is None:
		sliceN = mask1.array.shape[dim] // 2

	slices = [slice(None), slice(None), slice(None)]

	if mask1.getVolumeGeometryType() == PyIPSDK.eVGT_2d:
		slices.pop()
	else:
		slices[dim] = sliceN

	image1 = mask1.array
	image2 = mask2.array
	if image1.ndim > 2:
		image1 = np.squeeze(image1)
		image2 = np.squeeze(image2)
	if image1.ndim > 3:
		raise ValueError("image has too many non-singleton dimensions. Shape after squeezing: {}".format(image.shape))
	if image1.ndim > 2:
		if sliceN is None:
			sliceN = image1.shape[dim] // 2
		slices = [slice(None), slice(None), slice(None)]
		slices[dim] = sliceN
		image1 = image1[tuple(slices)]
		image2 = image2[tuple(slices)]

	fig = plt.figure(figsize=(8, 6))
	ax = plt.axes()

	cmap = ListedColormap(['black', 'white', 'red', 'blue'])
	bounds=[-0.5, 0.5, 1.5, 2.5, 3.5]
	norm = colors.BoundaryNorm(bounds, cmap.N)
 
	image1 = image1.astype(bool)
	image2 = image2.astype(bool)
 
	image = np.zeros(shape=image1.shape, dtype=int)
  
	common = np.logical_and(image1, image2)
	before = np.logical_and(image1, np.logical_not(common))
	after = np.logical_and(np.logical_not(common), image2)
	
	image[common] = 1
	image[before] = 2
	image[after] = 3

	im = ax.imshow(image, interpolation='nearest', cmap=cmap, norm=norm, **kwargs)
 
	# Create custom legend
	patches = [
        mpatches.Patch(color='black', label='no mask'),
        mpatches.Patch(color='white', label='common'),
        mpatches.Patch(color='red', label='before'),
        mpatches.Patch(color='blue', label='after')
    ]
 
	ax.legend(handles=patches, loc='best')
	
	showToolTip = kwargs.get('showToolTip', False)
	if showToolTip:
		ax.format_coord = Formatter(im)
	plt.draw()
	fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # Remove all outer padding
	plt.show(block=False)
 

	return im

# def showSliceInteractive(img, lims=None, dim=0, **kwargs):
#     # Handle image input
#     image = np.squeeze(img.array)
#     if image.ndim != 3:
#         raise ValueError("Only 3D images are supported after squeezing.")

#     # Get number of slices in the chosen dimension
#     max_slice = image.shape[dim] - 1
#     slider_slice = widgets.IntSlider(min=0, max=max_slice, step=1, value=max_slice // 2, description="Slice")

#     # Intensity range sliders for contrast adjustment
#     min_val = float(np.min(image))
#     max_val = float(np.max(image))
#     slider_vmin = widgets.FloatSlider(min=min_val, max=max_val, step=(max_val - min_val) / 100,
#                                       value=min_val if lims is None else lims[0], description="vmin")
#     slider_vmax = widgets.FloatSlider(min=min_val, max=max_val, step=(max_val - min_val) / 100,
#                                       value=max_val if lims is None else lims[1], description="vmax")

#     # Set up initial slice
#     def get_slice(index):
#         slices = [slice(None)] * 3
#         slices[dim] = index
#         return image[tuple(slices)]

#     # Create figure once
#     fig, ax = plt.subplots(figsize=(8, 6))
#     initial_img = get_slice(slider_slice.value)
#     imshow_args = {'cmap': kwargs.get('cmap', 'gray'),
#                    'vmin': slider_vmin.value, 'vmax': slider_vmax.value}
#     img_disp = ax.imshow(initial_img, **imshow_args)
#     ax.set_title(f"Slice {slider_slice.value}")
#     plt.tight_layout()
#     fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
#     plt.show()

#     # Update slice callback
#     def update_slice(change):
#         new_slice = slider_slice.value
#         img_disp.set_data(get_slice(new_slice))
#         ax.set_title(f"Slice {new_slice}")
#         fig.canvas.draw_idle()

#     # Update contrast callback
#     def update_contrast(change):
#         img_disp.set_clim(vmin=slider_vmin.value, vmax=slider_vmax.value)
#         fig.canvas.draw_idle()

#     # Link sliders
#     slider_slice.observe(update_slice, names='value')
#     slider_vmin.observe(update_contrast, names='value')
#     slider_vmax.observe(update_contrast, names='value')

#     # Display UI
#     ui = widgets.VBox([
#         slider_slice,
#         widgets.HBox([slider_vmin, slider_vmax])
#     ])
#     display(ui)


def showSliceInteractive(img, lims=None, dim=0, **kwargs):
    # Handle image input
    image = np.squeeze(img.array)
    if image.ndim != 3:
        raise ValueError("Only 3D images are supported after squeezing.")

    # Get number of slices in the chosen dimension
    max_slice = image.shape[dim] - 1
    slider_slice = widgets.IntSlider(
        min=0, max=max_slice, step=1, value=max_slice // 2, description="Slice"
    )

    # Intensity range slider (combined vmin/vmax)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    slider_vrange = widgets.FloatRangeSlider(
        value=[min_val if lims is None else lims[0],
               max_val if lims is None else lims[1]],
        min=min_val,
        max=max_val,
        step=(max_val - min_val) / 100,
        description="vmin/vmax",
        continuous_update=True
    )

    # Get slice function
    def get_slice(index):
        slices = [slice(None)] * 3
        slices[dim] = index
        return image[tuple(slices)]

    # Display UI above plot
    ui = widgets.VBox([
        slider_slice,
        slider_vrange
    ])
    display(ui)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    initial_img = get_slice(slider_slice.value)
    vmin, vmax = slider_vrange.value
    img_disp = ax.imshow(initial_img, cmap=kwargs.get('cmap', 'gray'),
                         vmin=vmin, vmax=vmax)
    ax.set_title(f"Slice {slider_slice.value}")
    plt.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()

    # Update callbacks
    def update_slice(change):
        img_disp.set_data(get_slice(slider_slice.value))
        ax.set_title(f"Slice {slider_slice.value}")
        fig.canvas.draw_idle()

    def update_contrast(change):
        vmin, vmax = slider_vrange.value
        img_disp.set_clim(vmin=vmin, vmax=vmax)
        fig.canvas.draw_idle()

    # Link sliders
    slider_slice.observe(update_slice, names='value')
    slider_vrange.observe(update_contrast, names='value')


def showSliceBeforeAfterInteractive(mask1, mask2, dim=0, **kwargs):
	image1 = np.squeeze(mask1.array).astype(bool)
	image2 = np.squeeze(mask2.array).astype(bool)

	if image1.ndim != 3 or image2.ndim != 3:
		raise ValueError("Both images must be 3D after squeezing.")

	if image1.shape != image2.shape:
		raise ValueError("Both masks must have the same shape.")

	max_slice = image1.shape[dim] - 1
	slider = widgets.IntSlider(min=0, max=max_slice, step=1, value=max_slice // 2, description="Slice")

	def get_slice(index):
		slices = [slice(None)] * 3
		slices[dim] = index
		img1 = image1[tuple(slices)]
		img2 = image2[tuple(slices)]

		common = np.logical_and(img1, img2)
		before = np.logical_and(img1, ~common)
		after = np.logical_and(img2, ~common)

		result = np.zeros_like(img1, dtype=np.uint8)
		result[common] = 1     # white
		result[before] = 2     # red
		result[after] = 3      # blue
		return result

	# Set up plot
	fig, ax = plt.subplots(figsize=(8, 6))
	cmap = ListedColormap(['black', 'white', 'red', 'blue'])
	bounds=[-0.5, 0.5, 1.5, 2.5, 3.5]
	norm = colors.BoundaryNorm(bounds, cmap.N)
	img_plot = ax.imshow(get_slice(slider.value), cmap=cmap, norm=norm, interpolation='nearest', **kwargs)
	title = ax.set_title(f"Slice {slider.value}")

	# Legend
	patches = [
		mpatches.Patch(color='black', label='No mask'),
		mpatches.Patch(color='white', label='Common'),
		mpatches.Patch(color='red', label='Before'),
		mpatches.Patch(color='blue', label='After')
	]
	ax.legend(handles=patches, loc='upper right')
	plt.tight_layout()
	fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # Remove all outer padding
	plt.show()

	def update(change):
		slice_index = change['new']
		img_plot.set_data(get_slice(slice_index))
		title.set_text(f"Slice {slice_index}")
		fig.canvas.draw_idle()

	slider.observe(update, names='value')
	display(slider)
 
 

def showOverlayInteractive(gray_image, binary_mask, dim=0, mask_color='red', initial_alpha=0.4, **kwargs):
    # Convert to arrays and squeeze
    gray = np.squeeze(gray_image.array)
    mask = np.squeeze(binary_mask.array).astype(np.uint8)  # ensure numeric 0/1

    # Check dimensions
    if gray.ndim != 3 or mask.ndim != 3:
        raise ValueError("Both images must be 3D after squeezing.")
    if gray.shape != mask.shape:
        raise ValueError("Grayscale image and binary mask must have the same shape.")

    max_slice = gray.shape[dim] - 1
    slider_slice = widgets.IntSlider(min=0, max=max_slice, step=1, value=max_slice // 2, description="Slice")

    # Intensity range for grayscale image (single range slider)
    min_val = float(np.min(gray))
    max_val = float(np.max(gray))
    slider_vrange = widgets.FloatRangeSlider(
        value=[min_val, max_val],
        min=min_val,
        max=max_val,
        step=(max_val - min_val) / 100,
        description="vmin/vmax",
        continuous_update=True
    )

    # Alpha slider for transparency
    slider_alpha = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=initial_alpha, description="Alpha")

    def get_slice(index):
        slices = [slice(None)] * 3
        slices[dim] = index
        return gray[tuple(slices)], mask[tuple(slices)]

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 6))
    gray_slice, mask_slice = get_slice(slider_slice.value)

    # Grayscale layer
    img_gray = ax.imshow(gray_slice, cmap='gray', interpolation='nearest',
                         vmin=slider_vrange.value[0], vmax=slider_vrange.value[1], **kwargs)

    # Mask colormap (transparent for 0, solid for 1)
    cmap = colors.ListedColormap([(0, 0, 0, 0), colors.to_rgba(mask_color)])
    img_mask = ax.imshow(mask_slice, cmap=cmap, alpha=slider_alpha.value, interpolation='nearest')

    title = ax.set_title(f"Slice {slider_slice.value}")
    ax.axis('off')
    plt.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Update functions
    def update_slice(change):
        index = slider_slice.value
        gray_slice, mask_slice = get_slice(index)
        img_gray.set_data(gray_slice)
        img_mask.set_data(mask_slice)
        title.set_text(f"Slice {index}")
        fig.canvas.draw_idle()

    def update_alpha(change):
        img_mask.set_alpha(slider_alpha.value)
        fig.canvas.draw_idle()

    def update_contrast(change):
        vmin, vmax = slider_vrange.value
        img_gray.set_clim(vmin=vmin, vmax=vmax)
        fig.canvas.draw_idle()

    # Connect sliders
    slider_slice.observe(update_slice, names='value')
    slider_alpha.observe(update_alpha, names='value')
    slider_vrange.observe(update_contrast, names='value')

    # Display
    ui = widgets.VBox([
        slider_slice,
        slider_vrange,
        slider_alpha
    ])
    display(ui)
    plt.show()
 

# def showOverlayInteractive(gray_image, binary_mask, dim=0, mask_color='red', initial_alpha=0.4, **kwargs):
#     # Convert to arrays and squeeze
#     gray = np.squeeze(gray_image.array)
#     mask = np.squeeze(binary_mask.array).astype(np.uint8)  # ensure numeric 0/1

#     # Check dimensions
#     if gray.ndim != 3 or mask.ndim != 3:
#         raise ValueError("Both images must be 3D after squeezing.")
#     if gray.shape != mask.shape:
#         raise ValueError("Grayscale image and binary mask must have the same shape.")

#     max_slice = gray.shape[dim] - 1
#     slider_slice = widgets.IntSlider(min=0, max=max_slice, step=1, value=max_slice // 2, description="Slice")

#     # Intensity range for grayscale image
#     min_val = float(np.min(gray))
#     max_val = float(np.max(gray))
#     slider_vmin = widgets.FloatSlider(min=min_val, max=max_val, step=(max_val - min_val) / 100,
#                                       value=min_val, description="vmin")
#     slider_vmax = widgets.FloatSlider(min=min_val, max=max_val, step=(max_val - min_val) / 100,
#                                       value=max_val, description="vmax")

#     # Alpha slider for transparency
#     slider_alpha = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=initial_alpha, description="Alpha")

#     def get_slice(index):
#         slices = [slice(None)] * 3
#         slices[dim] = index
#         return gray[tuple(slices)], mask[tuple(slices)]

#     # Set up plot
#     fig, ax = plt.subplots(figsize=(8, 6))
#     gray_slice, mask_slice = get_slice(slider_slice.value)

#     # Grayscale layer
#     img_gray = ax.imshow(gray_slice, cmap='gray', interpolation='nearest',
#                          vmin=slider_vmin.value, vmax=slider_vmax.value, **kwargs)

#     # Mask colormap (transparent for 0, solid for 1)
#     cmap = colors.ListedColormap([(0, 0, 0, 0), colors.to_rgba(mask_color)])

#     # Dummy mask for init
#     dummy_mask = np.zeros_like(mask_slice, dtype=np.uint8)
#     dummy_mask[0, 0] = 1  # ensures min/max present
#     img_mask = ax.imshow(dummy_mask, cmap=cmap,
#                          alpha=slider_alpha.value, interpolation='nearest')
#     img_mask.set_data(mask_slice)

#     title = ax.set_title(f"Slice {slider_slice.value}")
#     ax.axis('off')
#     plt.tight_layout()
#     fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

#     # Update functions
#     def update_slice(change):
#         index = slider_slice.value
#         gray_slice, mask_slice = get_slice(index)
#         img_gray.set_data(gray_slice)
#         img_mask.set_data(mask_slice)  # no re-cast, already uint8
#         title.set_text(f"Slice {index}")
#         fig.canvas.draw_idle()

#     def update_alpha(change):
#         img_mask.set_alpha(slider_alpha.value)
#         fig.canvas.draw_idle()

#     def update_contrast(change):
#         img_gray.set_clim(vmin=slider_vmin.value, vmax=slider_vmax.value)
#         fig.canvas.draw_idle()

#     # Connect sliders
#     slider_slice.observe(update_slice, names='value')
#     slider_alpha.observe(update_alpha, names='value')
#     slider_vmin.observe(update_contrast, names='value')
#     slider_vmax.observe(update_contrast, names='value')

#     # Display
#     ui = widgets.VBox([
#         slider_slice,
#         widgets.HBox([slider_vmin, slider_vmax]),
#         slider_alpha
#     ])
#     display(ui)
#     plt.show()

############################################################################################################
# functions to work on binned volumes or part of the volume



def get_crop_parameters_from_component_mask(components, bin_factor=1):
	#input: surface component contains all the different surface components as connected components
		
	#get ROI from component mask
	
	# Create the infoset
	inMeasureInfoSet3d = PyIPSDK.createMeasureInfoSet3d()
	PyIPSDK.createMeasureInfo(inMeasureInfoSet3d, "BoundingBoxMaxXMsr")
	PyIPSDK.createMeasureInfo(inMeasureInfoSet3d, "BoundingBoxMaxYMsr")
	PyIPSDK.createMeasureInfo(inMeasureInfoSet3d, "BoundingBoxMaxZMsr")
	
	PyIPSDK.createMeasureInfo(inMeasureInfoSet3d, "BoundingBoxMinXMsr")
	PyIPSDK.createMeasureInfo(inMeasureInfoSet3d, "BoundingBoxMinYMsr")
	PyIPSDK.createMeasureInfo(inMeasureInfoSet3d, "BoundingBoxMinZMsr")
	#Perform the analysis
	
	outMeasureSet = shapeanalysis.labelAnalysis3d(components, components, inMeasureInfoSet3d)
	# save results to csv format
	
	# retrieve measure results
	max_x = bin_factor*get_numpy_values(outMeasureSet.getMeasure("BoundingBoxMaxXMsr"))
	max_y = bin_factor*get_numpy_values(outMeasureSet.getMeasure("BoundingBoxMaxYMsr"))
	max_z = bin_factor*get_numpy_values(outMeasureSet.getMeasure("BoundingBoxMaxZMsr"))
	
	min_x = np.maximum(bin_factor * get_numpy_values(outMeasureSet.getMeasure("BoundingBoxMinXMsr")), 0)
	min_y = np.maximum(bin_factor * get_numpy_values(outMeasureSet.getMeasure("BoundingBoxMinYMsr")), 0)
	min_z = np.maximum(bin_factor * get_numpy_values(outMeasureSet.getMeasure("BoundingBoxMinZMsr")), 0)

	
	# Create the crop_parameters list
	crop_parameter_list = [(int(min_x[i]), int(min_y[i]), int(min_z[i]), int(max_x[i] - min_x[i]), int(max_y[i] - min_y[i]), int(max_z[i] - min_z[i])) for i in range(len(min_x))]
	
	return crop_parameter_list

def get_padded_img(img, pad_size=50):
    
	# Get the size of the image
	xsz = img.getSizeX()
	ysz = img.getSizeY()
	zsz = img.getSizeZ()

	# Create a new padded image
	empty_img = PyIPSDK.createImage(img.getBufferType(), xsz + pad_size * 2, ysz + pad_size * 2, zsz + pad_size * 2)
	util.eraseImg(empty_img, 0)
 
	# Put the original image into the padded image
	padded_img = PyIPSDK.createImage(empty_img) #initialize 
	util.putROI3dImg(empty_img, img, pad_size, pad_size, pad_size, padded_img) 
 
	ROI_original_img = PyIPSDK.createImage(empty_img) #initialize
	img_ROI = bin.lightThresholdImg(img, 0)
	util.putROI3dImg(empty_img, img_ROI, pad_size, pad_size, pad_size, ROI_original_img)
	# Set the ROI inside the padded image to the original image
	return padded_img, ROI_original_img

def get_unpadded_img(padded_img, pad_size=50):
    
	# Get the size of the image
	xsz = padded_img.getSizeX()
	ysz = padded_img.getSizeY()
	zsz = padded_img.getSizeZ()

	crop_parameter_padding = (pad_size, pad_size, pad_size, xsz-2*pad_size, ysz-2*pad_size, zsz-2*pad_size)

	img = util.getROI3dImg(padded_img, *crop_parameter_padding)
 
	return img

def get_ROI(mask, crop_parameter, pad_size=50):
    
	mask_cropped = util.getROI3dImg(mask, *crop_parameter)
  
	xsz = mask_cropped.getSizeX()
	ysz = mask_cropped.getSizeY()
	zsz = mask_cropped.getSizeZ()

     
	mask_padded = PyIPSDK.createImage(PyIPSDK.eImageBufferType.eIBT_Binary,xsz+pad_size*2,ysz+pad_size*2,zsz+pad_size*2)
	util.eraseImg(mask_padded, 0)

	ROI_padded = PyIPSDK.createImage(mask_padded)
	util.putROI3dImg(mask_padded, mask_cropped, pad_size, pad_size, pad_size, ROI_padded) 
	mask_padded = ROI_padded

	return mask_padded

def put_ROI(mask, mask_cropped, crop_parameter, pad_size=50):
        
	xsz = mask_cropped.getSizeX()
	ysz = mask_cropped.getSizeY()
	zsz = mask_cropped.getSizeZ()
 	
 
	crop_parameter_padding = (pad_size, pad_size, pad_size, xsz-2*pad_size, ysz-2*pad_size, zsz-2*pad_size)
 
	mask_cropped_unpadded = util.getROI3dImg(mask_cropped, *crop_parameter_padding)
	
	mask_with_ROI = PyIPSDK.createImage(mask)

	_ = util.putROI3dImg(mask, mask_cropped_unpadded, *crop_parameter[:3], mask_with_ROI) 

	return mask_with_ROI



def work_on_binned_volume(mask, transformation_func, *transformation_args, bin_factor=8, dilation_after_transformation=1, **transformation_kwargs):

	# 3d zoom 
	xsz = mask.getSizeX()
	ysz = mask.getSizeY()
	zsz = mask.getSizeZ()

	x_zoom = xsz / round(xsz/bin_factor)
	y_zoom = ysz / round(ysz/bin_factor)
	z_zoom = zsz / round(zsz/bin_factor)
	
	mask_binned = gtrans.zoom3dImg(mask, 1/x_zoom, 1/y_zoom, 1/z_zoom, PyIPSDK.eZoomInterpolationMethod.eZIM_NearestNeighbour)

	mask_transformed = transformation_func(mask_binned, *transformation_args, **transformation_kwargs)

	#dilate to compensate for binning
	if dilation_after_transformation is not None:
		structuringElement= PyIPSDK.sphericalSEXYZInfo(dilation_after_transformation)
		mask_transformed = morpho.dilate3dImg(mask_transformed, structuringElement)

	mask_transformed_unbinned = gtrans.zoom3dImg(mask_transformed, x_zoom, y_zoom, z_zoom,PyIPSDK.eZoomInterpolationMethod.eZIM_NearestNeighbour)
	
	mask_filtered = logic.bitwiseAndImgImg(mask, mask_transformed_unbinned)
	
	return mask_filtered

def bin_volume(img, bin_factor=8, target_img=None):
    
    # 3d zoom 
	xsz = img.getSizeX()
	ysz = img.getSizeY()
	zsz = img.getSizeZ()
 
	if target_img is None:

		x_zoom = xsz / round(xsz/bin_factor)
		y_zoom = ysz / round(ysz/bin_factor)
		z_zoom = zsz / round(zsz/bin_factor)
	
		img_binned = gtrans.zoom3dImg(img, 1/x_zoom, 1/y_zoom, 1/z_zoom, PyIPSDK.eZoomInterpolationMethod.eZIM_NearestNeighbour)
  
	else:
		x_zoom = xsz / target_img.getSizeX()
		y_zoom = ysz / target_img.getSizeY()
		z_zoom = zsz / target_img.getSizeZ()
	
		img_binned = gtrans.zoom3dImg(img, 1/x_zoom, 1/y_zoom, 1/z_zoom, PyIPSDK.eZoomInterpolationMethod.eZIM_NearestNeighbour)
 
	return img_binned

def unbin_volume(img_binned, bin_factor=8, target_img=None):
    
    # 3d zoom 
	xsz = target_img.getSizeX()
	ysz = target_img.getSizeY()
	zsz = target_img.getSizeZ()


 
	if target_img is None:
		x_zoom = xsz / round(xsz/bin_factor)
		y_zoom = ysz / round(ysz/bin_factor)
		z_zoom = zsz / round(zsz/bin_factor)
	else:
		x_zoom = xsz / img_binned.getSizeX()
		y_zoom = ysz / img_binned.getSizeY()
		z_zoom = zsz / img_binned.getSizeZ()
 
	
	img = gtrans.zoom3dImg(img_binned, x_zoom, y_zoom, z_zoom, PyIPSDK.eZoomInterpolationMethod.eZIM_NearestNeighbour)
 
	return img


# %%
