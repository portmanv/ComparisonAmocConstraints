import numpy as np
from sklearn.linear_model import LinearRegression

from functions import nanaverage

def compute_trend_3d_data(data_per_model_per_time_per_cell, times):
    (nb_models, nb_times, nb_cells) = data_per_model_per_time_per_cell.shape

    new_data = np.zeros((nb_models, nb_cells))
    
    reg = LinearRegression()
    
    for id_model in range(nb_models):
        # Pour le modèle, ne considère que les temps recouverts (pas de nan)
        times_to_keep = np.logical_not(np.sum(np.isnan(data_per_model_per_time_per_cell[id_model]), axis=1))
        x_true        = times[times_to_keep].astype('int').reshape(-1, 1)
        
        for id_cell in range(nb_cells):
            time_serie = data_per_model_per_time_per_cell[id_model, times_to_keep, id_cell]
            if np.sum(times_to_keep)<2:
                trend=np.nan
            else:
                # apprentissage sur 1 échantillon
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
        #name_Y = "AMOC slow-down % {}-{}\n(anomalie ref {}-{})".format(min_Y, max_Y, min_Y_ref, max_Y_ref)
        name_Y = "mean AMOC anomaly {}-{}\n(ref {}-{})".format(min_Y, max_Y, min_Y_ref, max_Y_ref)
        Y_simu = Y_simu_anom
    else:
        Y_simu = np.nanmean(Y[:, times_to_mean], axis=1)
        name_Y = "mean AMOC {}-{}".format(min_Y, max_Y)

    #print(np.nanmin(Y_ref), np.nanmax(Y_ref), np.nanmean(Y_ref), np.nanstd(Y_ref))
    #print(np.nanmin(Y_simu_anom), np.nanmax(Y_simu_anom))
    #print(np.nanmin(Y_simu), np.nanmax(Y_simu), np.nanmean(Y_simu), np.nanstd(Y_simu))
    
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


