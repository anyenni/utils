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


# TODO: Add scalebars to plotting functions


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

def showSliceInteractive(imgs, lims=None, dim=0, titles=None, **kwargs):
    """
    imgs: single image-like OR iterable of image-like objects with `.array`
    lims: (vmin, vmax) or None -> uses global range across all imgs
    dim: slicing dimension (0/1/2)
    titles: list of titles per image (optional)
    kwargs: passed to imshow (e.g. cmap='gray')
    """
    # --- normalize input to a list ---
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    # --- squeeze & validate ---
    volumes = []
    for img in imgs:
        vol = np.squeeze(img.array)
        if vol.ndim != 3:
            raise ValueError("Only 3D images are supported after squeezing.")
        volumes.append(vol)

    # --- require compatible shape for chosen dim (same slice index meaning) ---
    # simplest: require same shape along dim; you can relax if you want
    n_slices = [v.shape[dim] for v in volumes]
    if len(set(n_slices)) != 1:
        raise ValueError(f"All volumes must have the same number of slices along dim={dim}. Got {n_slices}")

    max_slice = n_slices[0] - 1

    # --- global min/max for shared contrast slider ---
    global_min = float(min(np.min(v) for v in volumes))
    global_max = float(max(np.max(v) for v in volumes))

    # --- widgets ---
    slider_slice = widgets.IntSlider(
        min=0, max=max_slice, step=1, value=max_slice // 2, description="Slice"
    )

    init_vmin = global_min if lims is None else lims[0]
    init_vmax = global_max if lims is None else lims[1]

    step = (global_max - global_min) / 100 if global_max > global_min else 1.0
    slider_vrange = widgets.FloatRangeSlider(
        value=[init_vmin, init_vmax],
        min=global_min,
        max=global_max,
        step=step,
        description="vmin/vmax",
        continuous_update=True
    )

    display(widgets.VBox([slider_slice, slider_vrange]))

    # --- slice helper ---
    def get_slice(volume, index):
        sl = [slice(None)] * 3
        sl[dim] = index
        return volume[tuple(sl)]

    # --- figure with N panels ---
    n = len(volumes)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), squeeze=False,  constrained_layout=True)
    axes = axes[0]

    cmap = kwargs.get("cmap", "gray")
    vmin, vmax = slider_vrange.value

    im_artists = []
    for i, (ax, vol) in enumerate(zip(axes, volumes)):
        im = ax.imshow(get_slice(vol, slider_slice.value), cmap=cmap, vmin=vmin, vmax=vmax)
        im_artists.append(im)

        if titles and i < len(titles):
            ax.set_title(f"{titles[i]} — Slice {slider_slice.value}")
        else:
            ax.set_title(f"Image {i+1} — Slice {slider_slice.value}")
        ax.axis("off")

    #plt.tight_layout()
    plt.show()

    # --- callbacks update all images ---
    def update_all(_change=None):
        idx = slider_slice.value
        vmin, vmax = slider_vrange.value

        for i, (im, vol, ax) in enumerate(zip(im_artists, volumes, axes)):
            im.set_data(get_slice(vol, idx))
            im.set_clim(vmin=vmin, vmax=vmax)

            if titles and i < len(titles):
                ax.set_title(f"{titles[i]} — Slice {idx}")
            else:
                ax.set_title(f"Image {i+1} — Slice {idx}")

        fig.canvas.draw_idle()

    slider_slice.observe(update_all, names="value")
    slider_vrange.observe(update_all, names="value")

# def showSliceInteractive(img, lims=None, dim=0, **kwargs):
#     # Handle image input
#     image = np.squeeze(img.array)
#     if image.ndim != 3:
#         raise ValueError("Only 3D images are supported after squeezing.")

#     # Get number of slices in the chosen dimension
#     max_slice = image.shape[dim] - 1
#     slider_slice = widgets.IntSlider(
#         min=0, max=max_slice, step=1, value=max_slice // 2, description="Slice"
#     )

#     # Intensity range slider (combined vmin/vmax)
#     min_val = float(np.min(image))
#     max_val = float(np.max(image))
#     slider_vrange = widgets.FloatRangeSlider(
#         value=[min_val if lims is None else lims[0],
#                max_val if lims is None else lims[1]],
#         min=min_val,
#         max=max_val,
#         step=(max_val - min_val) / 100,
#         description="vmin/vmax",
#         continuous_update=True
#     )

#     # Get slice function
#     def get_slice(index):
#         slices = [slice(None)] * 3
#         slices[dim] = index
#         return image[tuple(slices)]

#     # Display UI above plot
#     ui = widgets.VBox([
#         slider_slice,
#         slider_vrange
#     ])
#     display(ui)

#     # Create figure
#     fig, ax = plt.subplots(figsize=(8, 6))
#     initial_img = get_slice(slider_slice.value)
#     vmin, vmax = slider_vrange.value
#     img_disp = ax.imshow(initial_img, cmap=kwargs.get('cmap', 'gray'),
#                          vmin=vmin, vmax=vmax)
#     ax.set_title(f"Slice {slider_slice.value}")
#     plt.tight_layout()
#     fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
#     plt.show()

#     # Update callbacks
#     def update_slice(change):
#         img_disp.set_data(get_slice(slider_slice.value))
#         ax.set_title(f"Slice {slider_slice.value}")
#         fig.canvas.draw_idle()

#     def update_contrast(change):
#         vmin, vmax = slider_vrange.value
#         img_disp.set_clim(vmin=vmin, vmax=vmax)
#         fig.canvas.draw_idle()

#     # Link sliders
#     slider_slice.observe(update_slice, names='value')
#     slider_vrange.observe(update_contrast, names='value')


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
 
 
def showOverlayInteractive(
    gray_image,
    binary_mask=None,
    dim=0,
    mask_color="red",
    initial_alpha=0.4,
    use_clip_ignore_zeros=True,
    initial_mask_range=None,
    frame_px=3,                 # NEW: frame thickness in pixels
    frame_color=None,           # NEW: if None, uses mask_color
    **kwargs
):
    gray = np.squeeze(gray_image.array)

    mask = None
    if binary_mask is not None:
        mask = np.squeeze(binary_mask.array).astype(np.uint16)  # allow labels
        if gray.ndim != 3 or mask.ndim != 3:
            raise ValueError("Both images must be 3D after squeezing.")
        if gray.shape != mask.shape:
            raise ValueError("Grayscale image and binary mask must have the same shape.")
    else:
        if gray.ndim != 3:
            raise ValueError("gray_image must be 3D after squeezing.")

    # initial display vmin/vmax
    if use_clip_ignore_zeros:
        nz = gray > 0
        if np.any(nz):
            vmin, vmax = np.percentile(gray[nz], (1, 99))
        else:
            vmin, vmax = float(gray.min()), float(gray.max())
    else:
        vmin, vmax = float(gray.min()), float(gray.max())

    min_val, max_val = float(gray.min()), float(gray.max())

    max_slice = gray.shape[dim] - 1
    slider_slice = widgets.IntSlider(min=0, max=max_slice, step=1, value=max_slice // 2, description="Slice")

    slider_vrange = widgets.FloatRangeSlider(
        value=[float(vmin), float(vmax)],
        min=min_val,
        max=max_val,
        step=(max_val - min_val) / 100 if max_val > min_val else 1.0,
        description="vmin/vmax",
        continuous_update=True
    )

    slider_alpha = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=initial_alpha, description="Alpha")

    # Only used if no mask provided
    if binary_mask is None:
        if initial_mask_range is None:
            if use_clip_ignore_zeros and np.any(gray > 0):
                lo0, hi0 = np.percentile(gray[gray > 0], (10, 90))
            else:
                lo0, hi0 = np.percentile(gray, (10, 90))
            initial_mask_range = (float(lo0), float(hi0))

        slider_mask_range = widgets.FloatRangeSlider(
            value=[float(initial_mask_range[0]), float(initial_mask_range[1])],
            min=min_val,
            max=max_val,
            step=(max_val - min_val) / 200 if max_val > min_val else 1.0,
            description="Mask range",
            continuous_update=True
        )

    def get_slice(arr, index):
        s = [slice(None)] * 3
        s[dim] = index
        return arr[tuple(s)]

    def compute_mask_from_range(gray_slice):
        lo, hi = slider_mask_range.value
        return ((gray_slice >= lo) & (gray_slice <= hi)).astype(np.uint8)

    # NEW: small helper to add a colored frame to a 2D slice
    def add_frame(a2d, thickness, value):
        if thickness <= 0:
            return a2d
        out = a2d.copy()
        t = int(thickness)
        out[:t, :] = value
        out[-t:, :] = value
        out[:, :t] = value
        out[:, -t:] = value
        return out

    # initial slice
    idx0 = slider_slice.value
    gray_slice = get_slice(gray, idx0)

    if mask is not None:
        mask_slice = get_slice(mask, idx0)
    else:
        mask_slice = compute_mask_from_range(gray_slice)

    # NEW: determine if mask is binary (0/1) or label (0..N)
    is_label_mask = False
    if mask is not None:
        u = np.unique(mask_slice)
        # label mask if it contains integers beyond {0,1}
        is_label_mask = np.any((u != 0) & (u != 1))

    # NEW: build a robust label colormap (good for <= 5 labels, safe beyond)
    # First five are colorblind-friendly-ish and very distinguishable.
    base = [
        (0.90, 0.10, 0.10, 1.0),  # red
        (0.10, 0.60, 0.10, 1.0),  # green
        (0.10, 0.35, 0.90, 1.0),  # blue
        (0.95, 0.65, 0.10, 1.0),  # orange
        (0.60, 0.20, 0.80, 1.0),  # purple
    ]
    def label_cmap_for(max_label):
        # index 0 is fully transparent background
        if max_label <= 0:
            cols = [(0, 0, 0, 0)]
        else:
            cols = [(0, 0, 0, 0)] + [base[(i - 1) % len(base)] for i in range(1, max_label + 1)]
        return colors.ListedColormap(cols)

    fig, ax = plt.subplots()
    img_gray = ax.imshow(
        gray_slice, cmap="gray", interpolation="nearest",
        vmin=slider_vrange.value[0], vmax=slider_vrange.value[1], **kwargs
    )

    # NEW: frame (default color = mask_color)
    fc = frame_color if frame_color is not None else mask_color
    frame_rgba = colors.to_rgba(fc)
    # a separate overlay array: 0 = transparent, 1 = frame color
    frame_arr = add_frame(np.zeros_like(gray_slice, dtype=np.uint8), frame_px, 1)
    frame_cmap = colors.ListedColormap([(0, 0, 0, 0), frame_rgba])
    img_frame = ax.imshow(frame_arr, cmap=frame_cmap, interpolation="nearest", vmin=0, vmax=1)

    if mask is None or (mask is not None and not is_label_mask):
        # old behavior: binary mask (or computed from range)
        cmap = colors.ListedColormap([(0, 0, 0, 0), colors.to_rgba(mask_color)])
        img_mask = ax.imshow(mask_slice.astype(np.uint8), cmap=cmap, alpha=slider_alpha.value,
                             interpolation="nearest", vmin=0, vmax=1)
        mask_vmin, mask_vmax = 0, 1
    else:
        # NEW: label mask behavior
        max_label = int(mask_slice.max())
        cmap = label_cmap_for(max_label)
        img_mask = ax.imshow(mask_slice.astype(np.int32), cmap=cmap, alpha=slider_alpha.value,
                             interpolation="nearest", vmin=0, vmax=max_label)
        mask_vmin, mask_vmax = 0, max_label

    title = ax.set_title(f"Slice {idx0}")
    ax.axis("off")
    plt.tight_layout()

    def update_slice(change=None):
        idx = slider_slice.value
        g = get_slice(gray, idx)
        img_gray.set_data(g)

        # update frame to correct shape (in case shapes vary)
        f = add_frame(np.zeros_like(g, dtype=np.uint8), frame_px, 1)
        img_frame.set_data(f)

        if mask is not None:
            m = get_slice(mask, idx)
        else:
            m = compute_mask_from_range(g)

        img_mask.set_data(m)

        # NEW: if label mask, adapt colormap/vmax when labels change across slices
        if mask is not None:
            u = np.unique(m)
            label_now = np.any((u != 0) & (u != 1))
            if label_now:
                ml = int(m.max())
                img_mask.set_cmap(label_cmap_for(ml))
                img_mask.set_clim(vmin=0, vmax=ml)
            else:
                img_mask.set_cmap(colors.ListedColormap([(0, 0, 0, 0), colors.to_rgba(mask_color)]))
                img_mask.set_clim(vmin=0, vmax=1)

        title.set_text(f"Slice {idx}")
        fig.canvas.draw_idle()

    def update_alpha(change=None):
        img_mask.set_alpha(slider_alpha.value)
        fig.canvas.draw_idle()

    def update_contrast(change=None):
        vmin_, vmax_ = slider_vrange.value
        img_gray.set_clim(vmin=vmin_, vmax=vmax_)
        fig.canvas.draw_idle()

    slider_slice.observe(update_slice, names="value")
    slider_alpha.observe(update_alpha, names="value")
    slider_vrange.observe(update_contrast, names="value")
    if binary_mask is None:
        slider_mask_range.observe(update_slice, names="value")

    controls = [slider_slice, slider_vrange]
    if binary_mask is None:
        controls.append(slider_mask_range)
    controls.append(slider_alpha)

    display(widgets.VBox(controls))
    plt.show()
 

# def showOverlayInteractive(
#     gray_image,
#     binary_mask=None,
#     dim=0,
#     mask_color="red",
#     initial_alpha=0.4,
#     use_clip_ignore_zeros=True,
#     initial_mask_range=None,
#     **kwargs
# ):
#     gray = np.squeeze(gray_image.array)

#     mask = None
#     if binary_mask is not None:
#         mask = np.squeeze(binary_mask.array).astype(np.uint8)
#         if gray.ndim != 3 or mask.ndim != 3:
#             raise ValueError("Both images must be 3D after squeezing.")
#         if gray.shape != mask.shape:
#             raise ValueError("Grayscale image and binary mask must have the same shape.")
#     else:
#         if gray.ndim != 3:
#             raise ValueError("gray_image must be 3D after squeezing.")

#     # initial display vmin/vmax
#     if use_clip_ignore_zeros:
#         nz = gray > 0
#         if np.any(nz):
#             vmin, vmax = np.percentile(gray[nz], (1, 99))
#         else:
#             vmin, vmax = float(gray.min()), float(gray.max())
#     else:
#         vmin, vmax = float(gray.min()), float(gray.max())

#     min_val, max_val = float(gray.min()), float(gray.max())

#     max_slice = gray.shape[dim] - 1
#     slider_slice = widgets.IntSlider(min=0, max=max_slice, step=1, value=max_slice // 2, description="Slice")

#     slider_vrange = widgets.FloatRangeSlider(
#         value=[float(vmin), float(vmax)],
#         min=min_val,
#         max=max_val,
#         step=(max_val - min_val) / 100 if max_val > min_val else 1.0,
#         description="vmin/vmax",
#         continuous_update=True
#     )

#     slider_alpha = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=initial_alpha, description="Alpha")

#     # Only used if no mask provided
#     if binary_mask is None:
#         if initial_mask_range is None:
#             if use_clip_ignore_zeros and np.any(gray > 0):
#                 lo0, hi0 = np.percentile(gray[gray > 0], (10, 90))
#             else:
#                 lo0, hi0 = np.percentile(gray, (10, 90))
#             initial_mask_range = (float(lo0), float(hi0))

#         slider_mask_range = widgets.FloatRangeSlider(
#             value=[float(initial_mask_range[0]), float(initial_mask_range[1])],
#             min=min_val,
#             max=max_val,
#             step=(max_val - min_val) / 200 if max_val > min_val else 1.0,
#             description="Mask range",
#             continuous_update=True
#         )

#     def get_slice(arr, index):
#         s = [slice(None)] * 3
#         s[dim] = index
#         return arr[tuple(s)]

#     def compute_mask_from_range(gray_slice):
#         lo, hi = slider_mask_range.value
#         return ((gray_slice >= lo) & (gray_slice <= hi)).astype(np.uint8)

#     # initial slice
#     idx0 = slider_slice.value
#     gray_slice = get_slice(gray, idx0)
#     if mask is not None:
#         mask_slice = get_slice(mask, idx0)
#     else:
#         mask_slice = compute_mask_from_range(gray_slice)

#     fig, ax = plt.subplots()
#     img_gray = ax.imshow(
#         gray_slice, cmap="gray", interpolation="nearest",
#         vmin=slider_vrange.value[0], vmax=slider_vrange.value[1], **kwargs
#     )

#     cmap = colors.ListedColormap([(0, 0, 0, 0), colors.to_rgba(mask_color)])
#     img_mask = ax.imshow(mask_slice, cmap=cmap, alpha=slider_alpha.value,
#                          interpolation="nearest", vmin=0, vmax=1)

#     title = ax.set_title(f"Slice {idx0}")
#     ax.axis("off")
#     plt.tight_layout()

#     def update_slice(change=None):
#         idx = slider_slice.value
#         g = get_slice(gray, idx)
#         img_gray.set_data(g)

#         if mask is not None:
#             m = get_slice(mask, idx)
#         else:
#             m = compute_mask_from_range(g)
#         img_mask.set_data(m)

#         title.set_text(f"Slice {idx}")
#         fig.canvas.draw_idle()

#     def update_alpha(change=None):
#         img_mask.set_alpha(slider_alpha.value)
#         fig.canvas.draw_idle()

#     def update_contrast(change=None):
#         vmin_, vmax_ = slider_vrange.value
#         img_gray.set_clim(vmin=vmin_, vmax=vmax_)
#         fig.canvas.draw_idle()

#     slider_slice.observe(update_slice, names="value")
#     slider_alpha.observe(update_alpha, names="value")
#     slider_vrange.observe(update_contrast, names="value")
#     if binary_mask is None:
#         slider_mask_range.observe(update_slice, names="value")

#     controls = [slider_slice, slider_vrange]
#     if binary_mask is None:
#         controls.append(slider_mask_range)
#     controls.append(slider_alpha)

#     display(widgets.VBox(controls))
#     plt.show()

 
# def showOverlayInteractive(gray_image, binary_mask, dim=0, mask_color='red', initial_alpha=0.4, use_clip_ignore_zeros=True, **kwargs):
#     # Convert to arrays and squeeze
#     gray = np.squeeze(gray_image.array)
#     mask = np.squeeze(binary_mask.array).astype(np.uint8)  # ensure numeric 0/1

#     # Check dimensions
#     if gray.ndim != 3 or mask.ndim != 3:
#         raise ValueError("Both images must be 3D after squeezing.")
#     if gray.shape != mask.shape:
#         raise ValueError("Grayscale image and binary mask must have the same shape.")
    
#     # Compute initial vmin/vmax
#     if use_clip_ignore_zeros:
#         mask_nonzero = gray > 0
#         if np.any(mask_nonzero):
#             vmin, vmax = np.percentile(gray[mask_nonzero], (1, 99))
#         else:
#             vmin, vmax = float(np.min(gray)), float(np.max(gray))
#     else:
#         vmin, vmax = float(np.min(gray)), float(np.max(gray))

#     max_slice = gray.shape[dim] - 1
#     slider_slice = widgets.IntSlider(min=0, max=max_slice, step=1, value=max_slice // 2, description="Slice")

#     # Intensity range for grayscale image (single range slider)
#     min_val = float(np.min(gray))
#     max_val = float(np.max(gray))
#     slider_vrange = widgets.FloatRangeSlider(
#         value=[vmin, vmax],
#         min=min_val,
#         max=max_val,
#         step=(max_val - min_val) / 100 if max_val > min_val else 1.0,
#         description="vmin/vmax",
#         continuous_update=True
#     )

#     # Alpha slider for transparency
#     slider_alpha = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=initial_alpha, description="Alpha")

#     def get_slice(index):
#         slices = [slice(None)] * 3
#         slices[dim] = index
#         return gray[tuple(slices)], mask[tuple(slices)]
    
#     max_fig_size = 10
#     min_fig_size = 4

#     # --- Auto figure size based on slice dimensions ---
#     test_slice, _ = get_slice(slider_slice.value)
#     h, w = test_slice.shape
#     aspect_ratio = w / h

#     if aspect_ratio >= 1:  # wide image
#         fig_w = max_fig_size
#         fig_h = max(min_fig_size, max_fig_size / aspect_ratio)
#     else:  # tall image
#         fig_w = max(min_fig_size, max_fig_size * aspect_ratio)
#         fig_h = max_fig_size
        
#     fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
#     gray_slice, mask_slice = get_slice(slider_slice.value)

#     # Grayscale layer
#     img_gray = ax.imshow(
#         gray_slice, cmap='gray', interpolation='nearest',
#         vmin=slider_vrange.value[0], vmax=slider_vrange.value[1], **kwargs
#     )

#     # Mask colormap (transparent for 0, solid for 1)
#     cmap = colors.ListedColormap([(0, 0, 0, 0), colors.to_rgba(mask_color)])

#     # ⬇️ Fix: pin normalization so all-zero initial slices don't break it
#     img_mask = ax.imshow(
#         mask_slice, cmap=cmap, alpha=slider_alpha.value,
#         interpolation='nearest', vmin=0, vmax=1
#         # Alternatively: norm=colors.BoundaryNorm([-0.5, 0.5, 1.5], ncolors=2)
#     )

#     title = ax.set_title(f"Slice {slider_slice.value}")
#     ax.axis('off')
#     plt.tight_layout()
#     fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

#     # Update functions
#     def update_slice(change):
#         index = slider_slice.value
#         gray_slice, mask_slice = get_slice(index)
#         img_gray.set_data(gray_slice)
#         img_mask.set_data(mask_slice)
#         title.set_text(f"Slice {index}")
#         fig.canvas.draw_idle()

#     def update_alpha(change):
#         img_mask.set_alpha(slider_alpha.value)
#         fig.canvas.draw_idle()

#     def update_contrast(change):
#         vmin, vmax = slider_vrange.value
#         img_gray.set_clim(vmin=vmin, vmax=vmax)
#         fig.canvas.draw_idle()

#     # Connect sliders
#     slider_slice.observe(update_slice, names='value')
#     slider_alpha.observe(update_alpha, names='value')
#     slider_vrange.observe(update_contrast, names='value')

#     # Display
#     ui = widgets.VBox([slider_slice, slider_vrange, slider_alpha])
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
	
		img_binned = gtrans.zoom3dImg(img, 1/x_zoom, 1/y_zoom, 1/z_zoom, PyIPSDK.eZoomInterpolationMethod.eZIM_VolumeWeightedMean)
  
	else:
		x_zoom = xsz / target_img.getSizeX()
		y_zoom = ysz / target_img.getSizeY()
		z_zoom = zsz / target_img.getSizeZ()
	
		img_binned = gtrans.zoom3dImg(img, 1/x_zoom, 1/y_zoom, 1/z_zoom, PyIPSDK.eZoomInterpolationMethod.eZIM_VolumeWeightedMean)
 
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
 
	
	img = gtrans.zoom3dImg(img_binned, x_zoom, y_zoom, z_zoom, PyIPSDK.eZoomInterpolationMethod.eZIM_Linear)
 
	return img


# %%
