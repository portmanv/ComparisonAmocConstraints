import numpy as np

def import_scenario_AMOC(scenario, list_models_to_remove=[]):
    if scenario in ["ssp126", "ssp245", "ssp585"]:
        name_samples = np.load("../data/multiruns/name_samples_{}.npy".format(scenario))
    elif scenario in ["historical", "ssp119", "ssp370"]:
        name_run_per_sample   = np.load("../data/multiruns/{}_AMOC/name_run_per_model.npy".format(scenario))
        name_model_per_sample = np.load("../data/multiruns/{}_AMOC/name_model_per_model.npy".format(scenario))
        name_samples          = []
        for id_sample in range(len(name_run_per_sample)):
            name_samples.append(name_model_per_sample[id_sample]+'_'+name_run_per_sample[id_sample])
    else:
        raise NameError("{} inconnu".format(scenario))

        
    metric                = "AMOC"
    times                 = np.load("../data/multiruns/"+scenario+"_"+metric+"/times.npy").astype(str).astype(int)
    name_model_per_sample = np.load("../data/multiruns/"+scenario+"_"+metric+"/name_model_per_model.npy")
    name_run_per_sample   = np.load("../data/multiruns/"+scenario+"_"+metric+"/name_run_per_model.npy")
    nb_times              = len(times)
    nb_samples            = len(name_model_per_sample)
    full_name_samples     = [name_model_per_sample[id_sample]+"_"+name_run_per_sample[id_sample] for id_sample in range(nb_samples)]
    AMOC = np.memmap("../data/multiruns/"+scenario+"_"+metric+"/data.dat", dtype='float32', mode='r+',
                                  shape=(nb_samples, nb_times))
    
    # Take into account only common samples
    final_name_samples, id_samples_toget, _ = np.intersect1d(full_name_samples, name_samples, return_indices=True)
    AMOC = AMOC[id_samples_toget]
    name_models = name_model_per_sample[id_samples_toget]
    name_runs   = name_run_per_sample[id_samples_toget]
    
    times = times.astype('str').astype('int')
    
    return AMOC, times, name_models, name_runs, final_name_samples

def averagedRun_per_model(ssp_times, ssp_AMOC, ssp_name_samples, ssp_name_models):
    unique_models        = np.unique(ssp_name_models)
    ssp_AMOC_averagedRun = []
    for model in unique_models:
        ssp_AMOC_averagedRun.append(np.mean(ssp_AMOC[np.where(ssp_name_models==model)[0]], axis=0))
    return np.array(ssp_AMOC_averagedRun), unique_models

def singleRun_per_model(ssp_times, ssp_AMOC, ssp_name_samples, ssp_name_models):
    unique_models      = np.unique(ssp_name_models)
    ssp_AMOC_singleRun = []
    for model in unique_models:
        ssp_AMOC_singleRun.append(ssp_AMOC[np.where(ssp_name_models==model)[0][0]])
    return np.array(ssp_AMOC_singleRun), unique_models





