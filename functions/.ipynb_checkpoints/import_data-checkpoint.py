

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

from functions import display_map, nanaverage, nanstd
prop_cycle = plt.rcParams['axes.prop_cycle']



def import_CMIP6_TOS_SOS(scenario, years_to_select):
    list_name_per_var = ["tos", "sos"]

    nb_var = len(list_name_per_var)
    data_perVar           = []
    name_run_perVar       = []
    name_models_perVar    = []
    name_institute_perVar = []
    name_samples_perVar   = []

    # Import tos and then sos

    for var in list_name_per_var:
        data = np.load("../data/multiruns/"+"timeserie_"+var+"_"+scenario+"/data_perSample_perBox_perYear.npy")
        name_runs = np.load("../data/multiruns/"+"timeserie_"+var+"_"+scenario+"/name_run_perSample.npy")
        name_models = np.load("../data/multiruns/"+"timeserie_"+var+"_"+scenario+"/name_model_perSample.npy")
        name_institutes = np.load("../data/multiruns/"+"timeserie_"+var+"_"+scenario+"/name_institute_perSample.npy")
        nb_samples = len(name_runs)
        name_samples = [name_models[id_sample]+"_"+name_runs[id_sample] for id_sample in range(nb_samples)]
        data_perVar.append(data)
        name_run_perVar.append(name_runs)
        name_models_perVar.append(name_models)
        name_institute_perVar.append(name_institutes)
        name_samples_perVar.append(name_samples)
        (nb_samples, nb_box, nb_years) = data.shape

    # Check if the times are corrects
    if len(years_to_select)!=nb_years:
        raise NameError('pb')

    # Keep only the common model's runs 
    name_samples_X = reduce(np.intersect1d, name_samples_perVar)
    X_simu_perSample_perYear_perFeature_bis = np.zeros((len(name_samples_X), nb_years, nb_box*nb_var))
    id_feature = 0
    for id_box in range(nb_box):
        for id_var in range(nb_var):
            _, id_samples_out, id_samples_in = np.intersect1d(name_samples_X, name_samples_perVar[id_var], return_indices=True)
            X_simu_perSample_perYear_perFeature_bis[id_samples_out, :, id_feature] = data_perVar[id_var][id_samples_in, id_box, :]
            id_feature += 1

    return X_simu_perSample_perYear_perFeature_bis, name_samples_X, nb_var, list_name_per_var




def import_CMIP6_AMOC(scenario):
    
    # Import historical simulations
    metric = "AMOC"

    hist_times            = np.load("../data/multiruns/"+"historical"+"_"+metric+"_"+scenario+"/times.npy").astype(str).astype(int)
    name_model_per_sample = np.load("../data/multiruns/"+"historical"+"_"+metric+"_"+scenario+"/name_model_per_model.npy")
    name_run_per_sample   = np.load("../data/multiruns/"+"historical"+"_"+metric+"_"+scenario+"/name_run_per_model.npy")
    nb_times              = len(hist_times)
    nb_samples            = len(name_model_per_sample)
    hist_name_samples     = [name_model_per_sample[id_sample]+"_"+name_run_per_sample[id_sample] for id_sample in range(nb_samples)]
    hist_AMOC = np.memmap("../data/multiruns/"+"historical"+"_"+metric+"_"+scenario+"/data.dat", dtype='float32', mode='r+',
                                  shape=(nb_samples, nb_times))
    
    # Import future simulations
    period                = scenario
    ssp_times             = np.load("../data/multiruns/"+period+"_"+metric+"/times.npy").astype(str).astype(int)
    name_model_per_sample = np.load("../data/multiruns/"+period+"_"+metric+"/name_model_per_model.npy")
    name_run_per_sample   = np.load("../data/multiruns/"+period+"_"+metric+"/name_run_per_model.npy")
    nb_times              = len(ssp_times)
    nb_samples            = len(name_model_per_sample)
    ssp245_name_samples   = [name_model_per_sample[id_sample]+"_"+name_run_per_sample[id_sample] for id_sample in range(nb_samples)]
    ssp245_AMOC = np.memmap("../data/multiruns/"+period+"_"+metric+"/data.dat", dtype='float32', mode='r+',
                                  shape=(nb_samples, nb_times))
    
    return hist_times, hist_name_samples, hist_AMOC, ssp_times, ssp245_name_samples, ssp245_AMOC



def select_common_members(name_samples_X, hist_name_samples, ssp245_name_samples,
                         X_simu_perSample_perYear_perFeature_bis,
                         years_to_select, hist_times, ssp_times,
                         hist_AMOC, ssp245_AMOC):
    
    # Select only the run that are available for both TOS, SOS and AMOC
    final_name_samples = reduce(np.intersect1d, [name_samples_X, hist_name_samples, ssp245_name_samples])

    _, out_id_samples_X, in_id_samples_X       = np.intersect1d(final_name_samples, name_samples_X, return_indices=True)
    _, out_id_samples_hist, in_id_samples_hist = np.intersect1d(final_name_samples, hist_name_samples, return_indices=True)
    _, out_id_samples_ssp, in_id_samples_ssp   = np.intersect1d(final_name_samples, ssp245_name_samples, return_indices=True)

    X_simu_perSample_perYear_perFeature        = X_simu_perSample_perYear_perFeature_bis[in_id_samples_X]

    # Select only the times common between TOS, SOS and AMOC
    _, out_id_times_hist, in_id_times_hist = np.intersect1d(years_to_select, hist_times, return_indices=True)
    _, out_id_times_ssp, in_id_times_ssp   = np.intersect1d(years_to_select, ssp_times, return_indices=True)

    AMOC = np.nan*np.zeros((len(final_name_samples), len(years_to_select)))
    AMOC[:, out_id_times_hist] = np.copy(hist_AMOC[in_id_samples_hist][:, in_id_times_hist])
    # this is made possible by the fact that out_id_samples_hist is sorted
    AMOC[:, out_id_times_ssp]   = np.copy(ssp245_AMOC[in_id_samples_ssp][:, in_id_times_ssp])
    Y = np.copy(AMOC)
    times_Y = np.copy(years_to_select)

    return X_simu_perSample_perYear_perFeature, Y, times_Y, final_name_samples




import scipy.io
def import_AMOC_obs(display=True):
    # Import raw data
    path = "../data/last_obs_AMOC/moc_transports.mat"
    mat  = scipy.io.loadmat(path)
    
    obs_AMOC_times_days = mat['JG'][0]
    obs_AMOC_times_days = obs_AMOC_times_days.astype("datetime64[D]")-np.datetime64('1970-01-01')+np.datetime64('2004-04-01')
    obs_AMOC_times_years = obs_AMOC_times_days.astype("datetime64[Y]")
    AMOC_days = mat['MOC_mar_hc10'][0,:]
    
    # Yearly average
    mean_AMOC_perYear = []
    years_to_mean = np.arange(0,21).astype("datetime64[Y]")-np.datetime64('1970')+np.datetime64('2004')
    for year in years_to_mean:
        id_days = obs_AMOC_times_years==year
        sub_AMOC_days = AMOC_days[id_days]
        sub_AMOC_days = sub_AMOC_days[sub_AMOC_days>-99999.]
        mean_AMOC_perYear.append(np.mean(sub_AMOC_days))

    obs_AMOC_times  = np.array(years_to_mean).astype(str).astype(int)
    obs_AMOC_values = np.array(mean_AMOC_perYear)
    
    # Display time series
    if display:
        plt.scatter(obs_AMOC_times, obs_AMOC_values)
        plt.plot(obs_AMOC_times, obs_AMOC_values)
        plt.xticks(obs_AMOC_times[1::2])
        plt.title("Observed AMOC time serie")
        plt.ylabel("Intensity (Sv)")
        plt.xlabel("Year")
        plt.show()
    
    return obs_AMOC_times, obs_AMOC_values
    

def import_TOS_SOS_obs(years_to_select, display=True):
    nb_var = 2

    #------------------------------------------------------------ Import raw data
    obs_tos_times = np.load("../data/obs_SST_SSS/times.npy")
    obs_sos_times = obs_tos_times

    obs_tos_mask = np.load("../data/obs_SST_SSS/mask_tos_obs.npy")
    obs_sos_mask = np.load("../data/obs_SST_SSS/mask_sos_obs.npy")

    obs_tos_perYear_perCell     = np.load("../data/obs_SST_SSS/masked_tos_obs.npy")
    obs_sos_perYear_perCell     = np.load("../data/obs_SST_SSS/masked_sos_obs.npy")
    var_obs_tos_perYear_perCell = np.load("../data/obs_SST_SSS/masked_tos_obs_var.npy")
    var_obs_sos_perYear_perCell = np.load("../data/obs_SST_SSS/masked_sos_obs_var.npy")


    #------------------------------------------------------------ Display
    mean_obs_tos_perCell = np.mean(obs_tos_perYear_perCell, axis=0)
    mean_obs_sos_perCell = np.mean(obs_sos_perYear_perCell, axis=0)
    mM = np.max(np.abs(mean_obs_tos_perCell))
    if display:
        display_map(mean_obs_tos_perCell, obs_tos_mask, title="mean TOS obs", cmap='coolwarm',
                    vmin=-mM, vmax=mM)
        plt.show()

        display_map(mean_obs_sos_perCell, obs_sos_mask, title="mean SOS obs", cmap='coolwarm')
        plt.show()
    
    ObsData_mask_perVar            = [obs_tos_mask, obs_sos_mask]
    ObsData_perVar_perYear_perCell = [obs_tos_perYear_perCell, obs_sos_perYear_perCell]
    ObsVar_perVar_perYear_perCell  = [var_obs_tos_perYear_perCell, var_obs_sos_perYear_perCell]
    ObsTimes_perVar                = [obs_tos_times, obs_sos_times]
    
    
    
    #------------------------------------------------------------ Pooling the spatial masks and the times
    obs_times = np.arange(1900, 2022+1)

    mask_perVar  = []
    Times_perVar = []

    ObsData_perVar_perYear_perCell_ = [
        np.nan*np.zeros((len(obs_times), np.sum(ObsData_mask_perVar[id_var]))) for id_var in range(nb_var)]
    ObsVar_perVar_perYear_perCell_ = [
        np.nan*np.zeros((len(obs_times), np.sum(ObsData_mask_perVar[id_var]))) for id_var in range(nb_var)]

    for id_var in range(nb_var):

        #-------------- Find the shared spatial mask 
        mask_obs           = ObsData_mask_perVar[id_var]
        mask_perVar.append(mask_obs)

        #-------------- Find the shared times
        times, times_id_out_obs, times_id_in_obs   = np.intersect1d(obs_times, ObsTimes_perVar[id_var], return_indices=True)
        times, times_id_out_simu, times_id_in_simu = np.intersect1d(obs_times, years_to_select, return_indices=True)
        Times_perVar.append(times)

        #-------------- Storage of the intersection   
        ObsData_perVar_perYear_perCell_[id_var][times_id_out_obs] = ObsData_perVar_perYear_perCell[id_var][times_id_in_obs]
        ObsVar_perVar_perYear_perCell_[id_var][times_id_out_obs] = ObsVar_perVar_perYear_perCell[id_var][times_id_in_obs]



    return ObsData_perVar_perYear_perCell_, ObsVar_perVar_perYear_perCell_, obs_times, mask_perVar



def region_average_TOS_SOS_obs(longitudes, latitudes,
                               X_simu_perSample_perYear_perFeature,
                               ObsData_perVar_perYear_perCell_, ObsVar_perVar_perYear_perCell_,
                               area_perLat_per_Lon, list_name_per_var,
                               obs_times, mask_perVar, final_weight_per_sample, years_to_select,
                               display=False):

    #-------------------------------------- Definition of the spatial regions
    lonlat_lim_perBox = [[-30, 30, 65, 80],
                         [-70, -35, 45, 65],
                         [-40, 0, 45, 65],
                         [-90, 0, 15, 45],
                         [-70, 10, 0, 15],
                         [-50, 20, -15, 0],
                         [-60, 20, -40, -15],
                         [30, 110, -30, 20],
                         [-130, -90, -10, 10]]
    
    name_perBox = ["Nordic Seas", "Labrador", "Subpolar\nEast", "Subtropical",
                   "Tropical North", "Tropical South", "Atlantic South", "Indian Ocean", "Nino"]

    nb_box = len(name_perBox)
    if nb_box!= len(lonlat_lim_perBox):
        print("issue !")

    
    colors_perBox = prop_cycle.by_key()['color'][:nb_box]
    
    
    #-------------------------------------- Averaging for each region
    nb_var   = 2
    nb_times = len(obs_times)
    X_obs_perYear_perFeature           = np.nan*np.zeros((nb_times, nb_var*nb_box))
    X_obsVar_perYear_perFeature        = np.nan*np.zeros((nb_times, nb_var*nb_box))

    id_feature = 0
    list_id_box_perFeature = []
    list_id_var_perFeature = []
    list_idCell_perFeature = []
    middle_cell_perBox     = []
    list_name_perFeature   = []
    for id_box in range(nb_box):
        minLon, maxLon, minLat, maxLat = lonlat_lim_perBox[id_box]
        middle_cell_perBox.append([(minLon+maxLon)/2, (minLat+maxLat)/2])

        lon_to_keep = np.logical_and(longitudes>=minLon, longitudes<=maxLon)
        lat_to_keep = np.logical_and(latitudes>=minLat, latitudes<=maxLat)

        for id_var in range(nb_var):
            # Sélection des cellules correspondant à la zone
            ObsData_perYear_perCell_           = ObsData_perVar_perYear_perCell_[id_var]
            ObsVar_perYear_perCell_            = ObsVar_perVar_perYear_perCell_[id_var]

            mask_common = mask_perVar[id_var]
            idCells_to_keep = np.matmul(lat_to_keep.reshape(-1,1), lon_to_keep.reshape(1,-1))[mask_common]

            mask_region = np.logical_and(np.matmul(lat_to_keep.reshape(-1,1), lon_to_keep.reshape(1,-1)), mask_common)
            areas = area_perLat_per_Lon[mask_region]
            if display:
                print("{} cells".format(np.sum(mask_region)))

            X_obs_perYear_perFeature[:, id_feature] = np.average(
                                                  ObsData_perYear_perCell_[:, idCells_to_keep],
                                                  weights=areas, axis=1)

            # Variance de la somme (indépendance spatiale) = (somme variances)/N² = (moyenne)/N
            X_obsVar_perYear_perFeature[:, id_feature] = np.average(
                                                  ObsVar_perYear_perCell_[:, idCells_to_keep],
                                                  weights=areas, axis=1)/len(idCells_to_keep)

            list_id_box_perFeature.append(id_box)
            list_id_var_perFeature.append(id_var)
            list_idCell_perFeature.append(idCells_to_keep)
            list_name_perFeature.append("{} {}".format(list_name_per_var[id_var], name_perBox[id_box]))
            id_feature += 1
            
    
    # Display the time serie feature (SST or SSS for one region) per feature
    if display:
        (nb_samples, nb_times, nb_features) = X_simu_perSample_perYear_perFeature.shape
        for id_feature in range(nb_features):
            plt.figure()
            #plt.subplot(nb_features, 1, id_feature+1)

            mean_multi_model = nanaverage(X_simu_perSample_perYear_perFeature[:, :, id_feature], axis=0, weights=final_weight_per_sample)
            std_multi_model = nanstd(X_simu_perSample_perYear_perFeature[:, :, id_feature], axis=0, weights=final_weight_per_sample)
            plt.plot(years_to_select, mean_multi_model)
            plt.fill_between(years_to_select, mean_multi_model-std_multi_model, mean_multi_model+std_multi_model, alpha=0.5)
            plt.title("feature n°{}".format(id_feature))
            plt.show()

    return [X_obs_perYear_perFeature, X_obsVar_perYear_perFeature,
            name_perBox, list_idCell_perFeature, list_id_box_perFeature,
            list_id_var_perFeature, colors_perBox,
            middle_cell_perBox, list_name_perFeature, nb_box, lonlat_lim_perBox]


import matplotlib
from functions import reverse_mask
import cartopy.feature as cfeature
import cartopy.crs as ccrs

def display_regions(mask_perVar, name_perBox, list_idCell_perFeature, list_id_box_perFeature,
                    list_id_var_perFeature, name_cmap, latitudes, longitudes, middle_cell_perBox,
                    lonlat_lim_perBox):
    mask           = mask_perVar[0]
    nb_cells       = np.sum(mask)
    nb_box         = len(name_perBox)
    nb_features    = len(list_id_var_perFeature)
    id_box_perCell = np.zeros(nb_cells, dtype=int)

    for id_feature in np.flip(np.arange(nb_features)):
        if list_id_var_perFeature[id_feature]==0:
            id_cells = list_idCell_perFeature[id_feature]
            id_box   = list_id_box_perFeature[id_feature]
            id_box_perCell[id_cells] = id_box +1

    list_colors = list(matplotlib.colormaps[name_cmap].colors)[:nb_box]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cmap_name", ['#D3D3D3']+list_colors, N=nb_box+1)

    map_coefs = reverse_mask(id_box_perCell, np.logical_not(mask))
    nb_lat, nb_lon = map_coefs.shape 
    
    flipped_image      = np.flip(map_coefs, axis=0)
    flipped_latitudes  = np.flip(latitudes)
    translate_image    = np.concatenate((flipped_image[:, nb_lon//2:nb_lon], flipped_image[:, 0:nb_lon//2]), axis=1)
    flipped_longitudes = np.concatenate((longitudes[nb_lon//2:nb_lon], longitudes[0:nb_lon//2])).astype(int)


    plt.figure(figsize=(15*1.2,7*1.2), dpi=400)

    ax = plt.axes(projection=ccrs.PlateCarree())

    translate_image[np.isnan(translate_image)] = 0
    #ax.imshow(translate_image, interpolation='none', cmap=cmap)
    ax.pcolormesh(flipped_longitudes, flipped_latitudes, translate_image, cmap=cmap, # shading='auto', 
                             transform=ccrs.PlateCarree())

    if False:
        ax.set_xticks(ticks=np.arange(len(longitudes))[::20])
        ax.set_xticklabels(flipped_longitudes[::20])
        ax.set_yticks(ticks=np.arange(len(latitudes))[::30])
        ax.set_yticklabels(flipped_latitudes[::30])
    
        ax.set_xticks([])
        ax.set_yticks([])
    #ax.coastlines()
    #ax.add_feature(cfeature.LAND, facecolor="white")
    #ax.add_feature(cfeature.LAKE, facecolor="white")
    #ax.add_feature(cfeature.OCEAN,facecolor=("lightblue"))
    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=12, facecolor="white")


    #plt.ylabel("latitude", fontsize=None)
    #plt.xlabel("longitude", fontsize=None)

    displayed_lonlat = np.matmul(flipped_longitudes.reshape(-1,1), flipped_latitudes.reshape(1,-1))


    k = 0
    for id_feature in range(nb_features):
        if list_id_var_perFeature[id_feature]==0:
            id_box   = list_id_box_perFeature[id_feature]
            id_cells = list_idCell_perFeature[id_feature]
            color    = list_colors[id_box]
            [middle_lon, middle_lat] = middle_cell_perBox[id_box]
            rotation = 0
            if id_box==1:
                middle_lon += 3
                rotation = -33
            elif id_box==4:
                middle_lon -= 4
            elif id_box==5:
                middle_lon += 2
            xmin, xmax, ymin, ymax = np.copy(lonlat_lim_perBox[id_box])
            txt = ax.text(middle_lon, middle_lat, name_perBox[id_box], ha='center',va='center',
                     color="white", fontsize=15, zorder=1000, rotation=rotation) #, weight="bold"
            import matplotlib.patheffects as PathEffects
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
            
            if k%2==0: hatch='/'
            else: hatch='\\'

            if False:
                ax.add_patch(matplotlib.patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                                          hatch=hatch, fill=False, snap=False,
                                                          color="black", alpha=0.2))

            k += 1

    #plt.title("Definition of spatial regions", fontsize=18)#, weight="bold")
    plt.show()

