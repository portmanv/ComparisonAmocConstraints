# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:08:47 2022

@author: vport
"""

import numpy as np
import netCDF4 as nc
import pandas as pd
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





def nanaverage(data, weights=None, axis=None):
    
    masked_data = np.ma.masked_array(data, np.isnan(data))

    average = np.ma.average(masked_data, axis=axis, weights=weights)
    result = average.filled(np.nan)
    return result

def nanstd(data, weights=None, axis=None):
    
    masked_data = np.ma.masked_array(data, np.isnan(data))

    average = np.ma.average(masked_data, axis=axis, weights=weights)
    result = average.filled(np.nan)
    std = np.sqrt(np.ma.average((data-average)**2, axis=axis, weights=weights))
    return std





def display_number_of_runs(final_name_samples, display=True):
    nb_samples = len(final_name_samples)

    final_name_models = [final_name_samples[id_sample].split('_r')[0] for id_sample in range(nb_samples)]

    unique_models, counter = np.unique(final_name_models, return_counts=True)

    final_weight_per_sample = np.nan*np.zeros(nb_samples)
    for id_sample in range(nb_samples):
        nb_runs = counter[final_name_models[id_sample]==unique_models][0]
        final_weight_per_sample[id_sample] = 1/nb_runs

    data = np.concatenate((unique_models.reshape(-1,1), counter.reshape(-1,1)), axis=1)
    df = pd.DataFrame(data, columns=["name model", "number of runs"], index=np.arange(1,len(data)+1))
    if display:
        from IPython.display import display
        display(df)

    return final_name_models, final_weight_per_sample





