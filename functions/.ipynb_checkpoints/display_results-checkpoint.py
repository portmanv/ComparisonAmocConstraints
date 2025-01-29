import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

prop_cycle = plt.rcParams['axes.prop_cycle']



def plot_performances(name_methods, scenario, X_choice, anomalie_Y, name_Y,
                      list_list_predictions, list_list_std_without, list_list_LOO,
                      probability=0.90, return_LOOperFold=False, display_LOOperFold=False):
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



    #axes[0].errorbar(id_method+10, list_predictions, yerr=list_std, capsize=7, color=colors[id_method],
    #                         label="one predictor", fmt='o')
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

            if i==0:
                linestyle="solid"
                axes[1].fill_between(id_method+shift+np.array([-x,x]), y1=[xmin,xmin], y2=LOO_err*np.array([1,1]), color=colors[id_method], alpha=alpha)
            else:
                linestyle="dashed"
                axes[1].fill_between(id_method+shift+np.array([-x,x]), y1=[xmin,xmin], y2=LOO_err*np.array([1,1]),
                                     color="none", hatch="XXX", edgecolor=colors[id_method])
            if id_method==3 and i==1:
                axes[1].fill_between(id_method+shift+np.array([-x,x]), y1=xmin*np.ones(2), y2=(LOO_err)*np.ones(2),
                                 facecolors='none', edgecolors='black', linewidth=1)


            #axes[1].fill_between(id_method+shift+np.array([-x,x]), y1=[xmin,xmin], y2=LOO*np.array([1,1]), color=colors[id_method])

    axes[0].set_ylabel(name_Y+" (Sv)")
    axes[1].set_ylabel("Mean leave-one-out error")
    axes[1].set_xticks(range(nb_methods))
    axes[1].set_xticklabels(name_methods)
    axes[0].grid(axis="y")
    axes[1].grid(axis="y")
    axes[0].set_title("A. Constrained projections")
    axes[1].set_title("B. Cross-validation performances")
    plt.rc('axes', axisbelow=True)
    #axes[0].legend(title="Constrained by")
    leg = fig.legend([line1, line2, line3],
               ["one observed variable", "multiple observed variables", r"$\pm$"+" 90% empirical uncertainty"], #"1 "+r"$\sigma$"+" interval"],
               title="Constrained by", loc='center left', bbox_to_anchor=(0.9, 0.5))
    leg._legend_box.align = "left"
    #fig.suptitle("Comparison of method results")
    axes[1].get_xticklabels()[3].set_fontweight("bold")
    axes[1].set_ylim(0)
    plt.show()



def define_color_and_marker(common_models, list_front_markers, list_back_markers,
                            based_on_institute=True, other_names=[], other_models=[]):
    
    
    # Couleurs
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Instituts
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

    #------------- Affichage des incertitudes sur X_obs et X_simu
    if option==1: plt.subplot(121)
    else: plt.subplot(211)
    plt.rc('hatch', color='green', linewidth=0.1)
    plt.axvline(X_obs_AMOC, color='green', alpha=0.5, label='real-world observation', linewidth=2)
    plt.rc('hatch', color='red', linewidth=0.25)
    if False:
        plt.axvline(np.mean(X_simu_AMOC), color='red', alpha=0.5)
        line2 = plt.fill_betweenx([np.min(Y_simu), np.max(Y_simu)], np.mean(X_simu_AMOC)-np.std(X_simu_AMOC),
                      np.mean(X_simu_AMOC)+np.std(X_simu_AMOC),
                      alpha=0., hatch="//", label='multi-model mean $\pm \sigma$', linewidth=12)
    #line3 = plt.axhline(y=uni_wA_pred, linestyle='dotted', color='black',
    #                   label="prediction ({} Sv)".format(np.round(ypred_obs, 1)))
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    xmin, xmax = plt.gca().get_xlim()
    plt.errorbar(xmin+xmargin, uni_wA_pred, yerr=std_err_wA, capsize=7, color="black", lw=2, alpha=0.8, fmt="_",
                 label="{:.1f} ".format(uni_wA_pred)+r"$\pm$"+" {:.1f} Sv".format(std_err_wA)) # r"$\hat{f}(x_0)\pm \hat{\sigma}_{\varepsilon}$")
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #legend1 = plt.legend(handles=[line1, line3]) #line2
    legend1 = plt.legend()
    plt.ylabel(name_Y+" (Sv)")
    plt.xlabel(name_X_AMOC+" (Sv)")
    plt.title("A. Weighted average")

    
    
    #------------- Affichage des modèles
    list_front_markers = ["<", ">", "^", "v", "o", "s", "p", "*", "h", "H", "d", "P", "X"] #, "+", "x", "*", "o"
    list_back_markers  = ["o", "s", "d"]


    #common_models, id_common_AMOC_models, _ = np.intersect1d(
    #        AMOC_data_name_models, final_name_models, return_indices=True)
    list_color, list_front_marker, list_back_marker = define_color_and_marker(final_name_models, list_front_markers, list_back_markers)
    #list_color, list_front_marker, list_back_marker = np.array(list_color)[id_common_AMOC_models], np.array(list_front_marker)[id_common_AMOC_models], np.array(list_back_marker)[id_common_AMOC_models]


    list_points = []
    for id_model in range(len(final_name_models)):
        front_marker = list_front_marker[id_model]
        back_marker  = list_back_marker[id_model]
        color        = list_color[id_model]
        alpha        = 0.5
        a, = plt.plot(X_simu_AMOC[id_model], Y_simu[id_model], marker=back_marker, c=color, alpha=alpha, markersize=10, linewidth=0.)
        b, = plt.plot(X_simu_AMOC[id_model], Y_simu[id_model], marker=front_marker, label=final_name_models[id_model], c=color, alpha=1, linewidth=0.)
        list_points.append((a,b))

    #legend2 = plt.legend(list_points, common_models, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

    plt.gca().add_artist(legend1)

    plt.scatter(X_simu_AMOC, Y_simu, facecolors='none', s=300, edgecolors='black', linewidth=10*weights)



    if option==1: plt.subplot(122)
    else: plt.subplot(212)
    #------------- Affichage de la régression
    lr        = LinearRegression().fit(X_simu_AMOC.reshape(-1, 1), Y_simu)
    ypred     = lr.predict(X_simu_AMOC.reshape(-1, 1))
    ypred_obs = lr.predict(X_obs_AMOC.reshape(-1, 1))[0]
    x         = np.linspace(X_simu_AMOC.min(), X_simu_AMOC.max())
    y         = lr.predict(x.reshape(-1, 1))

    plt.axvline(X_obs_AMOC, color='green', alpha=0.5, label="real-world observation", linewidth=2)
    plt.rc('hatch', color='green', linewidth=0.25)
    #line1 = plt.fill_betweenx([np.min(Y_simu), np.max(Y_simu)], X_obs_AMOC-X_obs_AMOC_std, X_obs_AMOC+X_obs_AMOC_std,
    #                  alpha=0., hatch=r"\\", label='observation $\pm \sigma$')
    line2 = plt.plot(X_simu_AMOC, ypred, color='red')#, label='linear regression')
    plt.fill_between(x, y-std_err_LR, y+std_err_LR, color='red', alpha=0.2)

    xmin, xmax = plt.gca().get_xlim()
    plt.errorbar(xmin+xmargin, uni_lr_pred, yerr=std_err_LR, capsize=7, color="black", lw=2, alpha=0.8, fmt="_",
                 label="{:.1f} ".format(uni_lr_pred)+r"$\pm$"+" {:.1f} Sv".format(std_err_LR)) # r"$\hat{f}(x_0)\pm \hat{\sigma}_{\varepsilon}$")

    legend1 = plt.legend()
    plt.hlines(y=uni_lr_pred, xmin=xmin+xmargin, xmax=X_obs_AMOC, linestyle="dotted", color='black', linewidth=1)
    plt.hlines(y=uni_lr_pred-std_err_LR, xmin=xmin+xmargin, xmax=X_obs_AMOC, linestyle="dotted", color='black')
    plt.hlines(y=uni_lr_pred+std_err_LR, xmin=xmin+xmargin, xmax=X_obs_AMOC, linestyle="dotted", color='black')


    #------------- Affichage des modèles
    list_front_markers = ["<", ">", "^", "v", "o", "s", "p", "*", "h", "H", "d", "P", "X"] #, "+", "x", "*", "o"
    list_back_markers  = ["o", "s", "d"]

    #common_models, id_common_AMOC_models, _ = np.intersect1d(
    #        AMOC_data_name_models, final_name_models, return_indices=True)
    list_color, list_front_marker, list_back_marker = define_color_and_marker(final_name_models, list_front_markers, list_back_markers)
    #list_color, list_front_marker, list_back_marker = np.array(list_color)[id_common_AMOC_models], np.array(list_front_marker)[id_common_AMOC_models], np.array(list_back_marker)[id_common_AMOC_models]


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

    #legend2 = plt.legend(list_points, final_name_models, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    plt.ylabel(name_Y+" (Sv)")
    plt.xlabel(name_X_AMOC+" (Sv)")
    plt.title("B. Linear regression")
    plt.tight_layout()
    plt.show()