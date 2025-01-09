# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:08:47 2022

@author: vport
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


def load_lat_lon_area(path):
    ds = nc.Dataset(path)
    latitudes = ds["lat"][:].data
    longitudes = ds["lon"][:].data
    area_perLat_per_Lon = ds["cell_area"][:].data
    return latitudes, longitudes, area_perLat_per_Lon



def reverse_mask(array, mask, fill_value=np.nan):
    demasked_array                       = np.empty(mask.shape)
    demasked_array[:]                    = fill_value
    demasked_array[np.logical_not(mask)] = array
    
    return demasked_array


def display_map(vector, anti_spatial_mask, title="", cmap=None, vmin=None, vmax=None, return_image=False,
               colorbar_=True, ax=False, fontsize=None, display=True, fraction=0.15, remove_axes=False,
               cmap_ticks=None):
    map_coefs = reverse_mask(vector, np.logical_not(anti_spatial_mask))
    nb_lat, nb_lon = map_coefs.shape

    flipped_image   = np.flip(map_coefs, axis=0)
    translate_image = np.concatenate((flipped_image[:, nb_lon//2:nb_lon], flipped_image[:, 0:nb_lon//2]), axis=1)

    if display:
        if ax==False:
            plt.figure(figsize=(15,7))

            plt.imshow(translate_image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
            plt.title(title, fontsize=fontsize)
            if colorbar_: plt.colorbar()

            if remove_axes:
                plt.xticks(ticks=[], labels=[])
                plt.yticks(ticks=[], labels=[])
            else:
                plt.ylabel("latitude", fontsize=fontsize)
                plt.xlabel("longitude", fontsize=fontsize)
        else:
            cax = ax.imshow(translate_image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
            ax.set_title(title, fontsize=fontsize)
            if colorbar_: 
                pcm = ax.pcolormesh(translate_image,
                            cmap=cmap, vmax=vmax, vmin=vmin)
                plt.colorbar(cax, ax=ax, fraction=fraction, ticks=cmap_ticks, norm=norm)

            if remove_axes:
                ax.set_xticks(ticks=[], labels=[])
                ax.set_yticks(ticks=[], labels=[])
            else:
                ax.set_ylabel("latitude", fontsize=fontsize)
                ax.set_xlabel("longitude", fontsize=fontsize)

    if return_image:
        return translate_image