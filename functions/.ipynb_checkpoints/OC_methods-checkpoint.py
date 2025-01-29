import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

prop_cycle = plt.rcParams['axes.prop_cycle']


# biais pour l'estimateur de variance - covariance
ddof = 0
def cross_cov(X,Y, ddof=0):
    n, p_X = X.shape
    n, p_Y = Y.shape
    
    C = np.zeros((p_X, p_Y))
    for i in range(p_X):
        for j in range(p_Y):
            c = np.cov(X[:, i], Y[:, j], ddof=ddof)[0,1]
            C[i,j] = c
            
    return C

class model_wAverage():
    def __init__(self, scale=True, display_error_perD_perS=False, **kwargs):
        self.name_model="Weighted Average"
        self.Cov_obs   = kwargs['Cov_obs']
        self.display_error_perD_perS = display_error_perD_perS
        
    def compute_weights(self, X_simu, X_obs, sigma_D, sigma_S, **kwargs):
        nb_models = len(X_simu)
        w         = np.zeros(nb_models)
        list_S2   = np.zeros(nb_models)
        for i in range(nb_models):
            D2_i  = np.linalg.norm(X_simu[i]-X_obs)

            for j in range(nb_models):
                list_S2[j] = np.linalg.norm(X_simu[i]-X_simu[j])
            w[i] = np.exp(-D2_i/(sigma_D**2)) / np.sum(
                            np.exp(-list_S2/(sigma_S**2)))
        w /= np.sum(w)
        return w

    def calibrate_parameters(self, X_simu, Y_simu, X_obs, nb_steps=10):
        # Compute the range of parameter exploration
        nb_models = len(X_simu)
        values_D  = np.array([np.sqrt(np.linalg.norm(X_simu[i]-X_obs)) for i in range(nb_models)])
        values_S  = np.array([[np.sqrt(np.linalg.norm(X_simu[i]-X_simu[j])) for i in range(nb_models)] for j in range(nb_models)])

        range_sigma_D = np.median(values_D)*np.linspace(0.2,2, nb_steps)
        range_sigma_S = np.median(values_S)*np.linspace(0.2,2, nb_steps)

        # COmpute cross validation error for each parameter possibility
        kf = KFold(n_splits=nb_models)
        error_perD_perS = np.nan*np.zeros((nb_steps, nb_steps))        
        for id_D in range(nb_steps):
            for id_S in range(nb_steps):
                errors = np.zeros(nb_models)
                for i, (train_index, test_index) in enumerate(kf.split(X_simu)):
                    X_train, Y_train = X_simu[train_index], Y_simu[train_index]
                    X_test, Y_test   = X_simu[test_index], Y_simu[test_index]
                    w = self.compute_weights(X_train, X_obs, range_sigma_D[id_D], range_sigma_S[id_S])
                    errors[i] = np.sqrt(np.mean(np.square(Y_test - np.average(Y_train, weights=w))))
                error_perD_perS[id_D, id_S] = np.mean(errors)

        # Find the optimal parameters
        id_min_D = np.where(error_perD_perS==np.min(error_perD_perS))[0][0]
        id_min_S = np.where(error_perD_perS==np.min(error_perD_perS))[1][0]
        sigma_D = range_sigma_D[id_min_D]
        sigma_S = range_sigma_S[id_min_S]
        
        if self.display_error_perD_perS:
            plt.imshow(error_perD_perS)
            plt.colorbar()
            plt.xticks(np.arange(nb_steps), labels=range_sigma_S.round(1))
            plt.yticks(np.arange(nb_steps), labels=range_sigma_D.round(1))
            plt.xlabel(r"$\sigma_S$")
            plt.ylabel(r"$\sigma_D$", rotation=0)
            print("optimal: {:.1f}, {:.1f}".format(sigma_D, sigma_S))

        return sigma_D, sigma_S

    
    def fit(self, X_simu, Y_simu, **kwargs):
        self.X_simu = X_simu
        self.Y_simu = Y_simu
        return self
        
    def predict(self, X_obs, **kwargs):
        self.sigma_D, self.sigma_S = self.calibrate_parameters(
            self.X_simu, self.Y_simu, X_obs)
        
        self.w = self.compute_weights(self.X_simu, X_obs, self.sigma_D, self.sigma_S)
        y_pred = np.average(self.Y_simu, weights=self.w)
        return y_pred
    
    def analytic_std(self, X_simu, Y_simu):
        return np.nan




class model_RL_noisy():
    def __init__(self, scale=True, **kwargs):
        self.name_model="Linear Regression"
        self.Cov_obs   = kwargs['Cov_obs']
        
    def fit(self, X_simu, Y_simu, **kwargs):
        ddof = 0
        self.mu_X   = np.mean(X_simu, axis=0)
        self.mu_Y   = np.mean(Y_simu)
        self.Cov_XX = np.cov(X_simu.T, ddof=ddof)
        self.Cov_XY = cross_cov(X_simu, Y_simu.reshape(-1,1), ddof=ddof)
        self.Cov_YX = self.Cov_XY.T
        self.Cov_YY = np.cov(Y_simu.T, ddof=ddof)
        self.b1     = np.matmul(self.Cov_YX, np.linalg.inv(self.Cov_XX + self.Cov_obs)).T
        self.b0     = self.mu_Y - np.dot(self.mu_X, self.b1)
        self.coef   = self.b1
        
        return self
        
    def predict(self, X_obs, **kwargs):
        return self.b0 + np.dot(X_obs, self.b1)
    
    def analytic_std(self, X_simu, Y_simu):
        epsilon = Y_simu - self.predict(X_simu)
        
        var = np.var(epsilon) + np.matmul(self.b1.T, np.matmul(self.Cov_obs, self.b1))
        return np.sqrt(var)[0,0]
    
    



class model_RidgeCV_noisy():
    def __init__(self, list_alphas=np.linspace(0.001, 10, 100), cv=None, display=False, **kwargs):
        self.name_model  ="LR-Ridge"
        self.Cov_obs     = kwargs['Cov_obs']
        self.list_alphas = list_alphas
        self.model = RidgeCV(alphas=list_alphas)

    def fit(self, X_simu, Y_simu):
        self.scaler = StandardScaler()
        X_simu_n    = self.scaler.fit_transform(X_simu)
        self.model.fit(X_simu_n, Y_simu)
        return self

    def predict(self, X_obs, **kwargs):
        X_obs_n = self.scaler.transform(X_obs)
        return self.model.predict(X_obs_n)

class model_MMM():
    def fit(self, X_simu, Y_simu):
        self.pred = np.mean(Y_simu)
        return self
        
    def predict(self, X_obs, **kwargs):
        return self.pred
    
class model_RF():
    def __init__(self, n_estimators=500, max_depth=None, max_features="sqrt", **kwargs):
        self.name_model = "RF"
        self.Cov_obs    = kwargs['Cov_obs']
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                           max_features=max_features, n_jobs=-2)
                
    def fit(self, X_simu, Y_simu, **kwargs):
        self.model.fit(X_simu, Y_simu)
        
        return self
        
    def predict(self, X_obs, **kwargs):
        y_pred = self.model.predict(X_obs)
        return y_pred
    
    def analytic_var(self, X_simu, Y_simu, X_obs):
        return np.nan


def LOO(X_simu, Y_simu, regression_model, display=False, return_LOOperFold=False):
    kf = KFold(n_splits=len(X_simu))
    RMSE_perFold = np.zeros(len(X_simu))
    for i, (train_index, test_index) in enumerate(kf.split(X_simu)):
        if display: print("{}/{}".format(i+1, len(X_simu)))
        X_train = X_simu[train_index]
        Y_train = Y_simu[train_index]
        X_test  = X_simu[test_index].reshape(1,-1)
        Y_test  = Y_simu[test_index]
        RMSE_perFold[i] = np.sqrt(np.mean(np.square(Y_test - regression_model.fit(X_train, Y_train).predict(X_test))))

    if return_LOOperFold:
        return RMSE_perFold
    else:
        return np.mean(RMSE_perFold)




def performances_methods(X_simu_AMOC, X_obs_AMOC, X_simu, X_obs, Y_simu, return_LOOperFold=False):
    X_obs_AMOC_std=0
    Cov_obs=0

    xx_univariate   = [X_simu_AMOC.reshape(-1,1), np.array(X_obs_AMOC).reshape(-1,1), np.square(X_obs_AMOC_std).reshape(-1,1)]
    xx_mutlivariate = [X_simu, X_obs, Cov_obs]

    list_list_predictions = []
    list_list_std_with    = []
    list_list_std_without = []
    list_list_LOO         = []
    poids_wA              = []
    
    name_methods = ["Multi-Model\nMean", "Weighted\nAverage", "Linear\nRegression", "Ridge\nRegression", "Random\nForest"]
    for xx in [xx_univariate, xx_mutlivariate]:
        X_simu_  = xx[0]
        X_obs_   = xx[1]
        Cov_obs_ = xx[2]

        print("MMM\n")
        MMM            = model_MMM()
        MMM_prediction = MMM.fit(X_simu,Y_simu).predict(X_obs.reshape(1,-1))
        MMM_RMSE       = np.sqrt(np.mean(np.square(Y_simu-MMM_prediction)))
        MMM_LOO        = LOO(X_simu_, Y_simu, model_MMM(), return_LOOperFold=return_LOOperFold)

        print("wAverage\n")
        wA = model_wAverage(Cov_obs=Cov_obs_)
        wA.fit(X_simu_,Y_simu)
        wA_prediction  = wA.predict(X_obs_.reshape(1,-1))
        poids_wA.append(np.copy(wA.w))
        #!!!!!! on prédit le modèle sachant que c'est l'observation, du coup on le prédit parfaitement ?
        wA_predictions = [wA.predict(X_simu_[id_model].reshape(1,-1)) for id_model in range(len(X_simu_))]
        wA_LOO         = LOO(X_simu_, Y_simu, model_wAverage(Cov_obs=Cov_obs_), return_LOOperFold=return_LOOperFold)
        wA.fit(X_simu_,Y_simu)
        wA_prediction  = wA.predict(X_obs_.reshape(1,-1))

        print("LR\n")
        lr = model_RL_noisy(Cov_obs=Cov_obs_, display=True).fit(X_simu_, Y_simu)
        lr_prediction = lr.predict(X_obs_.reshape(1,-1))[0,0]
        lr_LOO  = LOO(X_simu_, Y_simu, model_RL_noisy(Cov_obs=Cov_obs_), return_LOOperFold=return_LOOperFold)

        print("Ridge\n")
        ridge = model_RidgeCV_noisy(Cov_obs=Cov_obs_)
        ridge.fit(X_simu_, Y_simu)
        ridge_prediction = ridge.predict(X_obs_.reshape(1,-1))[0]
        ridge_LOO  = LOO(X_simu_, Y_simu, model_RidgeCV_noisy(Cov_obs=Cov_obs_), display=False, return_LOOperFold=return_LOOperFold)

        print("RF\n")
        RF = model_RF(Cov_obs=Cov_obs_)
        RF.fit(X_simu_, Y_simu)
        RF_prediction = RF.predict(X_obs_.reshape(1,-1))[0]
        RF_LOO = LOO(X_simu_, Y_simu, model_RF(Cov_obs=Cov_obs_), return_LOOperFold=return_LOOperFold)

        # RMSE pour tous, sans bruit d'obs
        if True:
            list_std_without = []
            for model in [model_MMM(), model_wAverage(Cov_obs=Cov_obs_),
                          model_RL_noisy(Cov_obs=Cov_obs_),
                          model_RidgeCV_noisy(Cov_obs=Cov_obs_),
                          model_RF(Cov_obs=Cov_obs_)]:
                model.fit(X_simu_, Y_simu)
                predictions = np.array([model.predict(X_simu_[id_model].reshape(1,-1)).flatten() for id_model in range(len(X_simu_))])

                RMSE = np.sqrt(np.mean(np.square(Y_simu.reshape(-1,1)-predictions)))
                list_std_without.append(RMSE)

        list_predictions = [MMM_prediction, wA_prediction, lr_prediction, ridge_prediction, RF_prediction]
        list_LOO         = [MMM_LOO, wA_LOO, lr_LOO, ridge_LOO, RF_LOO]

        list_list_predictions.append(list_predictions)
        list_list_std_without.append(list_std_without)
        list_list_LOO.append(list_LOO)
        
    return name_methods, list_list_predictions, list_list_std_without, list_list_LOO, poids_wA





