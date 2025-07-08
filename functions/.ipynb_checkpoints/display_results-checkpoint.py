import numpy as np
from scipy.stats import t
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from sklearn.linear_model import LinearRegression

from functions import reverse_mask
prop_cycle = plt.rcParams['axes.prop_cycle']

def plot_performances(name_methods, scenario, X_choice, anomalie_Y, name_Y,
                      list_list_predictions, list_list_std_without, list_list_LOO,
                      probability=0.90, return_LOOperFold=False, display_LOOperFold=False):
    """
    """
    
    z = t.interval(probability, np.inf, loc=0, scale=1)[1]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6,5), dpi=300)
    nb_methods = len(name_methods)

    alpha = 0.6
    beta = 0.1
    x = 0.1

    if scenario=='ssp245':
        ymin = 4
        ymax = 20.5
        if X_choice== "trend":
            ymax=20
    elif scenario=='ssp585':
        ymin = 4
        ymax = 20.5+3
    elif scenario=='ssp126':
        ymin = 4
        ymax = 20.5+3
    else:
        ymin = 4
        ymax = 20.5


    #------------ Analytique
    colors = prop_cycle.by_key()['color']
    for id_method in range(nb_methods):
        for i in range(2):
            if id_method==0: shift = 0
            else: shift = beta*(2*i-1)
            std_without = list_list_std_without[i][id_method]
            pred        = list_list_predictions[i][id_method]
            y1=ymin*np.ones(2)
            y2=(pred)*np.ones(2)

            if anomalie_Y:
                ymin=-12
                ymax=0
                y2=0*np.ones(2)
                y1=(pred)*np.ones(2)

            if id_method!=0:
                if i==0:
                    linestyle="solid"
                    marker="x"
                    axes[0].fill_between(id_method+shift+np.array([-x,x]), y1=y1, y2=y2,
                                     color=colors[id_method], alpha=alpha)
                else:
                    linestyle="dashed"
                    marker="o"
                    axes[0].fill_between(id_method+shift+np.array([-x,x]), y1=y1, y2=y2,
                                     color="none", hatch="XXX", edgecolor=colors[id_method])

            else:
                line0 = axes[0].fill_between(id_method+shift+np.array([-x,x]), y1=y1, y2=y2,
                                     facecolors='none', edgecolors=colors[id_method], linewidth=2)

            axes[0].errorbar(id_method+shift, pred, yerr=z*std_without, capsize=7, color="black", lw=2, alpha=0.8, fmt="_")

            if id_method==3 and i==1:
                axes[0].fill_between(id_method+shift+np.array([-x,x]), y1=y1, y2=y2,
                                 facecolors='none', edgecolors='black', linewidth=1)



    axes[0].set_ylim(ymin, ymax)

    std  = list_list_std_without[0][0]
    pred = list_list_predictions[0][0]
       
    line1 = axes[0].fill_between(id_method+np.array([-x,x])+10, y1=(pred-std)*np.ones(2), y2=(pred+std)**np.ones(2),
                                 color="gray", alpha=alpha, label="one predictor")
    
    line2 = axes[0].fill_between(id_method+np.array([-x,x])+10, y1=(pred-std)*np.ones(2), y2=(pred+std)**np.ones(2),
                                 color="none", hatch="XXX", edgecolor="gray", label="multiple predictor")
    line3 = axes[0].errorbar(id_method+shift+10, pred, yerr=z*std, capsize=7, color="black", lw=2, alpha=0.8, fmt="_")

    axes[0].set_xlim(-0.5, nb_methods-0.5)

    #------------ Empirique (validation croisée)
    xmin = 0
    for id_method in range(nb_methods):
        for i in range(2):
            if id_method==0: shift = 0
            else: shift = beta*(2*i-1)
            
            if return_LOOperFold:
                LOO_err = np.mean(list_list_LOO[i][id_method])
                if display_LOOperFold:
                    LOOperFold = list_list_LOO[i][id_method]
                    axes[1].scatter((id_method+shift)*np.ones(len(LOOperFold)), LOOperFold, color=colors[id_method], s=5)
            else:
                LOO_err = list_list_LOO[i][id_method]

            if id_method!=0:
                if i==0:
                    linestyle="solid"
                    axes[1].fill_between(id_method+shift+np.array([-x,x]), y1=[xmin,xmin], y2=LOO_err*np.array([1,1]), color=colors[id_method], alpha=alpha)
                else:
                    linestyle="dashed"
                    axes[1].fill_between(id_method+shift+np.array([-x,x]), y1=[xmin,xmin], y2=LOO_err*np.array([1,1]),
                                         color="none", hatch="XXX", edgecolor=colors[id_method])
            else:
                axes[1].fill_between(id_method+shift+np.array([-x,x]), y1=[xmin,xmin], y2=LOO_err*np.array([1,1]),
                                         facecolors='none', edgecolors=colors[id_method], linewidth=2)

                
            if id_method==3 and i==1:
                axes[1].fill_between(id_method+shift+np.array([-x,x]), y1=xmin*np.ones(2), y2=(LOO_err)*np.ones(2),
                                 facecolors='none', edgecolors='black', linewidth=1)


    axes[0].set_ylabel(name_Y+" (Sv)")
    axes[1].set_ylabel("Mean leave-one-out error")
    axes[1].set_xticks(range(nb_methods))
    axes[1].set_xticklabels(name_methods)
    axes[0].grid(axis="y")
    axes[1].grid(axis="y")
    axes[0].set_title("A. Constrained projections")
    axes[1].set_title("B. Cross-validation performances")
    plt.rc('axes', axisbelow=True)
    leg = fig.legend([line1, line2, line0, line3],
               ["one observed variable", "multiple observed variables", "Unconstrained", r"$\pm$"+" 90% empirical uncertainty"], #"1 "+r"$\sigma$"+" interval"],
               title="Constrained by", loc='center left', bbox_to_anchor=(0.9, 0.5))
    leg._legend_box.align = "left"
    axes[1].get_xticklabels()[3].set_fontweight("bold")
    axes[1].set_ylim(0)
    plt.show()



def define_color_and_marker(common_models, list_front_markers, list_back_markers,
                            based_on_institute=True, other_names=[], other_models=[]):
    
    
    # Colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Get name of the institutes for each climate model
    if based_on_institute:
        name_models    = np.load("../data/historical_AMOC/name_model_per_model.npy")
        name_institute = np.load("../data/historical_AMOC/name_institute_per_model.npy")
        
    else:
        name_models    = other_models
        name_institute = other_names
        
    _, indexes_to_keep, new_indexes = np.intersect1d(name_models, common_models, return_indices=True)
    institute_per_model   = name_institute[indexes_to_keep]
    institute_per_model[np.where(institute_per_model=='DKRZ')] = 'MPI-M'
    nb_models = len(common_models[new_indexes])

    unique_institute_per_model = []
    for institute in institute_per_model:
        if not (institute in unique_institute_per_model):
            unique_institute_per_model.append(institute)
    unique_institute_per_model = np.array(unique_institute_per_model)

    # 
    k = 0
    k_per_institute   = np.zeros(len(unique_institute_per_model), dtype=int)
    list_color        = []
    list_front_marker = []
    list_back_marker  = []
    for id_model in range(nb_models):
        institute    = institute_per_model[id_model]
        id_institute = np.where(unique_institute_per_model==institute)[0][0]
        color        = colors[id_institute%len(colors)]
        depassement  = id_institute//len(colors)        
        front_marker = list_front_markers[k_per_institute[id_institute]]
        back_marker  = list_back_markers[depassement]
        
        list_color.append(color)
        list_front_marker.append(front_marker)
        list_back_marker.append(back_marker)
        k_per_institute[id_institute] += 1
    
    return list_color, list_front_marker, list_back_marker


def display_univariate_WeightedAverage_LinearRegression(X_simu_AMOC, X_simu, Y_simu,
                                                        X_obs_AMOC, X_obs,
                                                        list_list_std_without, list_list_predictions,
                                                        name_Y, name_X_AMOC, final_name_models,
                                                        poids_wA, probability=0.90):
    
    # Values and uncertainties at a given probability
    z = t.interval(probability, np.inf, loc=0, scale=1)[1]
    std_err_LR  = z*list_list_std_without[0][2]
    std_err_wA  = z*list_list_std_without[0][1]
    weights     = poids_wA[0]
    uni_lr_pred = list_list_predictions[0][2]
    uni_wA_pred = list_list_predictions[0][1]

    # Display parameters
    alpha = 0.9
    option = 1 #1 ou 2
    xmargin = 0.5

    if option==1:
        fig = plt.figure(figsize=(12*alpha,4*alpha), dpi=300)
    else:
        fig = plt.figure(figsize=(6*alpha, 8*alpha), dpi=300)

    #------------- Display the uncertainties
    if option==1: plt.subplot(121)
    else: plt.subplot(211)
    plt.rc('hatch', color='green', linewidth=0.1)
    plt.axvline(X_obs_AMOC, color='green', alpha=0.5, label='real-world observation', linewidth=2)
    plt.rc('hatch', color='red', linewidth=0.25)
    xmin, xmax = plt.gca().get_xlim()
    plt.errorbar(xmin+xmargin, uni_wA_pred, yerr=std_err_wA, capsize=7, color="black", lw=2, alpha=0.8, fmt="_",
                 label="{:.1f} ".format(uni_wA_pred)+r"$\pm$"+" {:.1f} Sv".format(std_err_wA)) # r"$\hat{f}(x_0)\pm \hat{\sigma}_{\varepsilon}$")
    legend1 = plt.legend()
    plt.ylabel(name_Y+" (Sv)")
    plt.xlabel(name_X_AMOC+" (Sv)")
    plt.title("A. Weighted average")

    
    
    #------------- Display climate models
    list_front_markers = ["<", ">", "^", "v", "o", "s", "p", "*", "h", "H", "d", "P", "X"] #, "+", "x", "*", "o"
    list_back_markers  = ["o", "s", "d"]

    list_color, list_front_marker, list_back_marker = define_color_and_marker(final_name_models, list_front_markers, list_back_markers)

    list_points = []
    for id_model in range(len(final_name_models)):
        front_marker = list_front_marker[id_model]
        back_marker  = list_back_marker[id_model]
        color        = list_color[id_model]
        alpha        = 0.5
        a, = plt.plot(X_simu_AMOC[id_model], Y_simu[id_model], marker=back_marker, c=color, alpha=alpha, markersize=10, linewidth=0.)
        b, = plt.plot(X_simu_AMOC[id_model], Y_simu[id_model], marker=front_marker, label=final_name_models[id_model], c=color, alpha=1, linewidth=0.)
        list_points.append((a,b))

    plt.gca().add_artist(legend1)

    plt.scatter(X_simu_AMOC, Y_simu, facecolors='none', s=300, edgecolors='black', linewidth=10*weights)


    if option==1: plt.subplot(122)
    else: plt.subplot(212)
    #------------- Display the linear regression
    lr        = LinearRegression().fit(X_simu_AMOC.reshape(-1, 1), Y_simu)
    ypred     = lr.predict(X_simu_AMOC.reshape(-1, 1))
    ypred_obs = lr.predict(X_obs_AMOC.reshape(-1, 1))[0]
    x         = np.linspace(X_simu_AMOC.min(), X_simu_AMOC.max())
    y         = lr.predict(x.reshape(-1, 1))

    plt.axvline(X_obs_AMOC, color='green', alpha=0.5, label="real-world observation", linewidth=2)
    plt.rc('hatch', color='green', linewidth=0.25)
    line2 = plt.plot(X_simu_AMOC, ypred, color='red')#, label='linear regression')
    plt.fill_between(x, y-std_err_LR, y+std_err_LR, color='red', alpha=0.2)

    xmin, xmax = plt.gca().get_xlim()
    plt.errorbar(xmin+xmargin, uni_lr_pred, yerr=std_err_LR, capsize=7, color="black", lw=2, alpha=0.8, fmt="_",
                 label="{:.1f} ".format(uni_lr_pred)+r"$\pm$"+" {:.1f} Sv".format(std_err_LR)) # r"$\hat{f}(x_0)\pm \hat{\sigma}_{\varepsilon}$")

    legend1 = plt.legend()
    plt.hlines(y=uni_lr_pred, xmin=xmin+xmargin, xmax=X_obs_AMOC, linestyle="dotted", color='black', linewidth=1)
    plt.hlines(y=uni_lr_pred-std_err_LR, xmin=xmin+xmargin, xmax=X_obs_AMOC, linestyle="dotted", color='black')
    plt.hlines(y=uni_lr_pred+std_err_LR, xmin=xmin+xmargin, xmax=X_obs_AMOC, linestyle="dotted", color='black')


    #------------- Display climate models
    list_front_markers = ["<", ">", "^", "v", "o", "s", "p", "*", "h", "H", "d", "P", "X"] #, "+", "x", "*", "o"
    list_back_markers  = ["o", "s", "d"]

    list_color, list_front_marker, list_back_marker = define_color_and_marker(final_name_models, list_front_markers, list_back_markers)
    list_points = []
    for id_model in range(len(final_name_models)):
        front_marker = list_front_marker[id_model]
        back_marker  = list_back_marker[id_model]
        color        = list_color[id_model]
        alpha        = 0.5
        a, = plt.plot(X_simu_AMOC[id_model], Y_simu[id_model], marker=back_marker, c=color, alpha=alpha, markersize=10, linewidth=0.)
        b, = plt.plot(X_simu_AMOC[id_model], Y_simu[id_model], marker=front_marker, label=final_name_models[id_model], c=color, alpha=1, linewidth=0.)
        list_points.append((a,b))

    if option==1:
        fig.legend(list_points, final_name_models, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=6)
    else:
        fig.legend(list_points, final_name_models, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    plt.ylabel(name_Y+" (Sv)")
    plt.xlabel(name_X_AMOC+" (Sv)")
    plt.title("B. Linear regression")
    plt.tight_layout()
    plt.show()




def display_importance_per_feature(mean_importance_perfeature,
                                   mask_perVar, list_idCell_perFeature,
                                   list_id_box_perFeature, list_id_var_perFeature,
                                   latitudes, longitudes,
                                   display_AMOC=False, shrink=1.0, nb_slots_cmap=10, impose_mM=0):
    
    fontsize = 20
    
    nb_lon = len(longitudes)
    nb_lat = len(latitudes)
    nb_features = len(mean_importance_perfeature)
    indexes_sos = list(np.arange(1,nb_features, 2))
    indexes_tos = list(np.arange(0,nb_features, 2))

    mean_importance_perCell_tos = np.nan*np.zeros(np.sum(mask_perVar[0]))
    mean_importance_perCell_sos = np.nan*np.zeros(np.sum(mask_perVar[1]))

    for id_feature in range(nb_features-1):
        id_cells = list_idCell_perFeature[id_feature]
        id_box   = list_id_box_perFeature[id_feature]
        if list_id_var_perFeature[id_feature]==0:
            mean_importance_perCell_tos[id_cells] = mean_importance_perfeature[id_feature]
        elif list_id_var_perFeature[id_feature]==1:
            mean_importance_perCell_sos[id_cells] = mean_importance_perfeature[id_feature]

    map_coefs_tos = reverse_mask(mean_importance_perCell_tos, np.logical_not(mask_perVar[0]))
    map_coefs_sos = reverse_mask(mean_importance_perCell_sos, np.logical_not(mask_perVar[1]))

    mM = np.nanmax(np.abs(mean_importance_perfeature))
    mM = (np.ceil(10*mM)+(np.ceil(10*mM)%2))/10

    if impose_mM!=0:
        mM = impose_mM
    # Colormap
    cmap = plt.cm.coolwarm
    import matplotlib as mpl
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(-mM, mM, nb_slots_cmap)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    

    plt.figure()
    if display_AMOC:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 7), dpi=300, subplot_kw={'projection': ccrs.Robinson()})
    else:
        alpha = 1.8
        fig, (ax3, ax2) = plt.subplots(1,2, figsize=(alpha*12, alpha*3), dpi=300, subplot_kw={'projection': ccrs.Robinson()})
    
    im = ax2.pcolormesh(longitudes, latitudes, map_coefs_tos, cmap=cmap, # shading='auto', 
                             transform=ccrs.PlateCarree(), norm=norm)
    ax2.add_feature(cfeature.LAND, edgecolor='black', zorder=12)#, facecolor="white")
    ax2.set_title("B. Correction on future AMOC (Sv)\ninduced by the sea surface "+ r"$\bf{temperature}$", fontsize=20)
    
    im = ax3.pcolormesh(longitudes, latitudes, map_coefs_sos, cmap=cmap, # shading='auto', 
                             transform=ccrs.PlateCarree(), norm=norm)
    ax3.add_feature(cfeature.LAND, edgecolor='black', zorder=12)#, facecolor="white")

    map_coefs_sos_noNan = np.copy(map_coefs_sos)
    map_coefs_sos_noNan[np.isnan(map_coefs_sos_noNan)] = -1
    ax3.set_title("A. Correction on future AMOC (Sv)\ninduced by the sea surface "+ r"$\bf{salinity}$", fontsize=fontsize)
    
    if display_AMOC:
        im_bis = np.copy(translate_image)
        im_bis[np.logical_not(np.isnan(im_bis))] = -1
        ax1.pcolormesh(flipped_longitudes, flipped_latitudes, im_bis, cmap=cmap, # shading='auto', 
                             transform=ccrs.PlateCarree(), norm=norm)
        ax1.set_title("2005-2023 AMOC importance")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.get_ylim()
        id_ = np.argmin(np.abs(flipped_latitudes-26))
        #ax1.axhline(y=id_) #, c=mean_importance_perfeature[-1], cmap="coolwarm", vmin=-mM, vmax=mM)
        id_longitudes_kept = np.where(np.logical_and(flipped_longitudes>-80, flipped_longitudes<-15))[0]
        ax1.scatter(id_longitudes_kept,
                    np.repeat(id_, len(id_longitudes_kept)),
                    c=np.repeat(mean_importance_perfeature[-1], len(id_longitudes_kept)),
                    cmap="coolwarm", vmin=-mM, vmax=mM, s=5)

        ax1.set_yticks(ticks=[id_], labels=["26°N"])
        cbar = fig.colorbar(im, ax=[ax1, ax2, ax3], label="(Sv)", shrink=shrink, location='bottom')
    else:
        cbar = fig.colorbar(im, ax=[ax2, ax3], label="(Sv)", shrink=shrink, location='bottom')#, anchor=(0.5, 1.8))
    cbar.set_label(label='(Sv)', size=fontsize, rotation=0)
    cbar.ax.tick_params(labelsize=fontsize)
    
    plt.subplots_adjust(left=None, bottom=0.3, right=None, top=8, wspace=0.03, hspace=None)



