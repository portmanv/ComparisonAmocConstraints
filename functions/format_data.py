import numpy as np
from sklearn.linear_model import LinearRegression

from import_data import import_CMIP6_TOS_SOS, import_CMIP6_AMOC, select_common_members, import_AMOC_obs, import_TOS_SOS_obs, region_average_TOS_SOS_obs, display_regions
from functions import load_lat_lon_area, display_map, reverse_mask, nanaverage, nanstd, display_number_of_runs



def compute_trend_3d_data(data_per_model_per_time_per_cell, times):
    (nb_models, nb_times, nb_cells) = data_per_model_per_time_per_cell.shape

    new_data = np.zeros((nb_models, nb_cells))
    
    reg = LinearRegression()
    
    for id_model in range(nb_models):
        # Get rid of missing data (np.nan)
        times_to_keep = np.logical_not(np.sum(np.isnan(data_per_model_per_time_per_cell[id_model]), axis=1))
        x_true        = times[times_to_keep].astype('int').reshape(-1, 1)
        
        for id_cell in range(nb_cells):
            time_serie = data_per_model_per_time_per_cell[id_model, times_to_keep, id_cell]
            if np.sum(times_to_keep)<2:
                trend=np.nan
            else:
                # Learn on one sample
                y_true = time_serie.reshape(-1, 1)

                reg.fit(x_true, y_true)
                trend = reg.coef_[0,0]

            new_data[id_model, id_cell] = trend
            
    return new_data


def create_X_AMOC_feature(X_choice, Y, times_Y,
                          obs_AMOC_values, obs_AMOC_times,
                          min_X_AMOC=2005, max_X_AMOC=2023):

    times_to_mean  = np.logical_and(times_Y>=min_X_AMOC, times_Y<=max_X_AMOC)
    times_obs_AMOC = np.logical_and(obs_AMOC_times<=max_X_AMOC, obs_AMOC_times>=min_X_AMOC)

    # name_X 
    if X_choice=="mean":
        X_simu_AMOC = np.nanmean(Y[:, times_to_mean], axis=1)
        X_obs_AMOC  = np.mean(obs_AMOC_values[times_obs_AMOC])
        name_X_AMOC = "mean AMOC {}-{}".format(min_X_AMOC, max_X_AMOC)

    elif X_choice=="trend":
        nb_years    = len(times_Y[times_to_mean])
        nb_models   = len(Y)
        X_simu_AMOC = compute_trend_3d_data(Y[:, times_to_mean].reshape((nb_models, nb_years, 1)),
                                times_Y[times_to_mean])[:, 0]
        name_X_AMOC = "trend AMOC {}-{}".format(min_X_AMOC, max_X_AMOC)
        nb_years = len(obs_AMOC_times[times_obs_AMOC])
        X_obs_AMOC  = compute_trend_3d_data(obs_AMOC_values[times_obs_AMOC].reshape((1, nb_years, 1)),
                                obs_AMOC_times[times_obs_AMOC])[0, 0]


    X_AMOC_mean_1850_1900 = np.nanmean(Y[:,
                            np.logical_and(times_Y>=1850, times_Y<=1900)], axis=1)


    return name_X_AMOC, X_simu_AMOC, X_obs_AMOC, X_AMOC_mean_1850_1900



def create_X_TOS_SOS_features(X_choice, min_X, max_X, 
                              X_obs_perYear_perFeature, obs_times,
                              X_simu_perSample_perYear_perFeature, years_to_select):
    obs_times_to_keep = np.logical_and(min_X<=obs_times, obs_times<=max_X)
    simu_times_to_keep = np.logical_and(min_X<=years_to_select, years_to_select<=max_X)
    times    = years_to_select[simu_times_to_keep]
    nb_years = len(times)


    if X_choice=="mean":
        X_obs  = np.nanmean(X_obs_perYear_perFeature[obs_times_to_keep], axis=0)
        X_simu = np.nanmean(X_simu_perSample_perYear_perFeature[:, simu_times_to_keep], axis=1)
    elif X_choice=="trend":
        nb_years = len(obs_times[obs_times_to_keep])
        nb_features = len(X_obs_perYear_perFeature[0])
        X_obs  = compute_trend_3d_data(X_obs_perYear_perFeature[obs_times_to_keep].T.reshape((nb_features, nb_years, 1)),
                                times)[:, 0]
        X_simu = compute_trend_3d_data(X_simu_perSample_perYear_perFeature[:, simu_times_to_keep],
                                times)
        
    return X_obs, X_simu


def create_Y_AMOC_feature(anomalie_Y, min_Y, max_Y,
                          min_Y_ref, max_Y_ref,
                          Y, years_to_select):
    times_to_mean = np.logical_and(years_to_select>=min_Y, years_to_select<=max_Y)
    Y_ref  = nanaverage(Y[:, np.logical_and(years_to_select>=min_Y_ref, years_to_select<=max_Y_ref)], axis=1)
    if False:
        Y_anom_percent = 100 * (Y_ref.reshape(-1,1)-Y)/Y_ref.reshape(-1,1)
        Y_simu_anom = np.nanmean(Y_anom_percent[:, times_to_mean], axis=1)
    else:
        Y_anom  = Y - Y_ref.reshape(-1,1)
        Y_simu_anom = np.nanmean(Y_anom[:, times_to_mean], axis=1)

    if anomalie_Y:
        name_Y = "mean AMOC anomaly {}-{}\n(ref {}-{})".format(min_Y, max_Y, min_Y_ref, max_Y_ref)
        Y_simu = Y_simu_anom
    else:
        Y_simu = np.nanmean(Y[:, times_to_mean], axis=1)
        name_Y = "mean AMOC {}-{}".format(min_Y, max_Y)
        
    return Y_simu, name_Y, Y_ref



def average_member_perModel(final_name_models, X_simu, Y_simu,
                            Y_ref, X_simu_AMOC, X_AMOC_mean_1850_1900):

    uniques_models = np.unique(final_name_models)

    nb_models   = len(uniques_models)
    nb_features = len(X_simu[0])

    Y_ref_resampled       = np.nan*np.zeros(nb_models)
    Y_simu_resampled      = np.nan*np.zeros(nb_models)
    Y_simu_first_run      = np.nan*np.zeros(nb_models)
    X_simu_resampled      = np.nan*np.zeros((nb_models, nb_features))
    X_simu_first_run      = np.nan*np.zeros((nb_models, nb_features))
    X_simu_AMOC_resampled = np.nan*np.zeros(nb_models)
    X_AMOC_mean_1850_1900_resampled = np.nan*np.zeros(nb_models)
    id_samples_perModel   = []

    for id_model in range(nb_models):
        name_model = uniques_models[id_model]
        id_samples_to_average = np.where(final_name_models==name_model)[0]
        id_samples_perModel.append(id_samples_to_average)
        Y_simu_resampled[id_model] = np.mean(Y_simu[id_samples_to_average])
        Y_ref_resampled[id_model]  = np.mean(Y_ref[id_samples_to_average])
        X_simu_resampled[id_model] = np.mean(X_simu[id_samples_to_average], axis=0)
        X_simu_AMOC_resampled[id_model] = np.mean(X_simu_AMOC[id_samples_to_average])
        Y_simu_first_run[id_model] = Y_simu[id_samples_to_average[0]]
        X_simu_first_run[id_model] = X_simu[id_samples_to_average[0]]
        X_AMOC_mean_1850_1900_resampled[id_model] = X_AMOC_mean_1850_1900[id_samples_to_average[0]]
        
    return uniques_models, Y_simu_resampled, Y_ref_resampled, X_simu_resampled, X_simu_AMOC_resampled, X_AMOC_mean_1850_1900_resampled


def construct_data_given_scenario(scenario,
                                 X_choice, anomalie_Y, min_Y, max_Y,
                                 min_Y_ref, max_Y_ref, min_X, max_X,
                                 include_AMOC_in_predictors, return_all=False):
    #================================================ Import data
    #---- Import labels longitudes and latitudes
    latitudes, longitudes, area_perLat_per_Lon = load_lat_lon_area("../data/area_r360x180.nc")
    longitudes[longitudes>180] -= 360

    #---- Import CMIP6 TOS and SOS
    years_to_select = np.arange(1850, 2100+1)
    [X_simu_perSample_perYear_perFeature_bis, name_samples_X, nb_var, list_name_per_var
                ] = import_CMIP6_TOS_SOS(scenario, years_to_select)

    #---- Import CMIP6 AMOC
    [hist_times, hist_name_samples, hist_AMOC, ssp_times, ssp245_name_samples, ssp245_AMOC
            ] = import_CMIP6_AMOC(scenario)

    #---- Select only the members that are available for both TOS, SOS and AMOC
    [X_simu_perSample_perYear_perFeature, Y, times_Y, final_name_samples
            ] = select_common_members(name_samples_X, hist_name_samples, ssp245_name_samples,
                             X_simu_perSample_perYear_perFeature_bis,
                             years_to_select, hist_times, ssp_times,
                             hist_AMOC, ssp245_AMOC)

    #---- Import AMOC observations (RAPID)
    obs_AMOC_times, obs_AMOC_values = import_AMOC_obs(display=False)

    #---- Display the number of members given by each climate model
    final_name_models, final_weight_per_sample = display_number_of_runs(final_name_samples, display=False)
    
    #---- Import TOS and SOS observations
    [ObsData_perVar_perYear_perCell_, ObsVar_perVar_perYear_perCell_,
     obs_times, mask_perVar
            ] = import_TOS_SOS_obs(years_to_select, display=False)

    [X_obs_perYear_perFeature, X_obsVar_perYear_perFeature,
     name_perBox, list_idCell_perFeature, list_id_box_perFeature,
     list_id_var_perFeature, colors_perBox,
     middle_cell_perBox, list_name_perFeature, nb_box, lonlat_lim_perBox
            ] = region_average_TOS_SOS_obs(longitudes, latitudes, X_simu_perSample_perYear_perFeature,
                                   ObsData_perVar_perYear_perCell_, ObsVar_perVar_perYear_perCell_,
                                   area_perLat_per_Lon, list_name_per_var, obs_times, mask_perVar,
                                   final_weight_per_sample, years_to_select,
                                   display=False)
    
    #================================================ Construct X and Y
    #------- Construct X when it is univariate
    [name_X_AMOC, X_simu_AMOC, X_obs_AMOC, X_AMOC_mean_1850_1900
            ] = create_X_AMOC_feature(X_choice, Y, times_Y, obs_AMOC_values, obs_AMOC_times)

    #------- Construct X when it is multivariate
    X_obs, X_simu = create_X_TOS_SOS_features(X_choice, min_X, max_X, 
                                              X_obs_perYear_perFeature, obs_times,
                                              X_simu_perSample_perYear_perFeature, years_to_select)

    #------- Construct Y
    Y_simu, name_Y, Y_ref = create_Y_AMOC_feature(anomalie_Y, min_Y, max_Y,
                              min_Y_ref, max_Y_ref,
                              Y, years_to_select)

    #------- Keep only samples (climate models) that contains no Nan (some data are missing)
    model_to_keep     = np.logical_not(np.isnan(Y_simu))
    Y_ref             = Y_ref[model_to_keep]
    Y_simu            = Y_simu[model_to_keep]
    X_simu            = X_simu[model_to_keep]
    X_simu_AMOC       = X_simu_AMOC[model_to_keep]
    final_name_samples = final_name_samples[model_to_keep]
    final_name_models  = np.array(final_name_models)[model_to_keep]
    X_AMOC_mean_1850_1900   = X_AMOC_mean_1850_1900[model_to_keep]

    #------- When X is multivariate, it can contains the past AMOC (choice in the beginning of this notebook)
    if include_AMOC_in_predictors:
        X_simu = np.concatenate((X_simu, X_simu_AMOC.reshape(-1,1)), axis=1)
        X_obs  = np.concatenate((X_obs, X_obs_AMOC.reshape(-1)))
        list_name_perFeature.append(name_X_AMOC)
        list_id_var_perFeature.append(2)

    #------- Average each model on its available members
    [uniques_models, Y_simu_resampled, Y_ref_resampled, X_simu_resampled, X_simu_AMOC_resampled, X_AMOC_mean_1850_1900_resampled
            ] = average_member_perModel(final_name_models, X_simu, Y_simu,
                                Y_ref, X_simu_AMOC, X_AMOC_mean_1850_1900)
    nb_models = len(uniques_models)

    #------- Rename the final variables
    X_simu                = X_simu_resampled
    Y_simu                = Y_simu_resampled
    Y_ref                 = Y_ref_resampled
    X_simu_AMOC           = X_simu_AMOC_resampled
    X_AMOC_mean_1850_1900 = X_AMOC_mean_1850_1900_resampled
    final_name_models     = np.copy(uniques_models)

    if return_all:
        print("{} (RAPID) is {:.2f} Sv.".format(name_X_AMOC, X_obs_AMOC))
        return [X_simu, Y_simu, Y_ref, X_simu_AMOC,
                X_AMOC_mean_1850_1900, final_name_models,
                X_obs, X_obs_AMOC, name_X_AMOC, name_Y,
                mask_perVar, name_perBox, list_idCell_perFeature,
                list_id_box_perFeature, list_id_var_perFeature,
                latitudes, longitudes, middle_cell_perBox, lonlat_lim_perBox, list_name_perFeature]
    else:
        return [X_simu, Y_simu, Y_ref, X_simu_AMOC,
                X_AMOC_mean_1850_1900, final_name_models,
                X_obs, X_obs_AMOC, name_X_AMOC, name_Y]


        

