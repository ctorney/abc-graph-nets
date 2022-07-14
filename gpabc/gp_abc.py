
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn import linear_model
from sklearn.metrics import r2_score


import GPy
from math import *
from scipy.stats import qmc
from tqdm import tqdm

class abcGP:
    def __init__(self, halton_start, halton_range, input_dim, n_points,T, simulator, likelihood_function, max_l=None, min_l=None):

        self.halton_start = halton_start
        self.halton_range = halton_range
        self.input_dim = input_dim
        self.n_points = n_points
        self.l_init = halton_range/((n_points-1)**(1./input_dim))
        self.T = T
        self.halton_generator = qmc.Halton(d=self.input_dim, scramble=False)
        self.halton_generator.fast_forward(1) # skip the first point at origin
        self.halton_points = self.halton_generator.random(self.n_points)
        self.skip = self.n_points

        self.halton_points = self.halton_start + self.halton_range*self.halton_points
        self.simulator = simulator

        self.likelihood_function = likelihood_function

        self.gp = []
        self.sim_output = [None]*self.n_points
        self.likelihood = np.full(self.n_points, np.nan)
        #self.reg_coeff = np.random.normal(size=(self.input_dim,nss))

    def runWave(self):
        # run a GP wave

        for i, p in tqdm(enumerate(self.halton_points)):

            # points have a nan likelihood if they haven't been evaluated
            # so we skip points that are finite to include previous sims in the GP
            if not np.isnan(self.likelihood[i]):
                continue
            self.sim_output[i] = self.simulator(p)
            self.likelihood[i] = self.likelihood_function(self.sim_output[i])#,self.reg_coeff)



        # fit gp
        Y = self.likelihood[np.isfinite(self.likelihood)]

        X = self.halton_points[np.isfinite(self.likelihood)]
        Y_mean = np.mean(Y)
        Y_std = np.std(Y)+1e-3

        Y = np.atleast_2d((Y-Y_mean)/Y_std).T

        # Instantiate a Gaussian Process model
        #kernel = ConstantKernel() * RBF(length_scale = self.l_init, length_scale_bounds=(self.min_l, self.max_l)) + WhiteKernel( noise_level_bounds=(1e-05, 0.1))
        ##length_scale=[1] * num_features
        kernel = GPy.kern.RBF(input_dim= self.input_dim, ARD=True, variance=1., lengthscale=self.l_init)

        #gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5) #9   
        m = GPy.models.GPRegression(X,Y,kernel)
        
        # calculate the resolution of points per dimension
        points_per_dim = ((len(self.gp)+1)*self.n_points)**(1.0/self.input_dim)


        # constrain the lengthscale so that the minimum L is greater than the approx distance between points
        for i in range(self.input_dim):
            minL = self.halton_range[i]/points_per_dim
            maxL = self.halton_range[i]

            m.rbf.lengthscale[[i]].constrain_bounded(minL,maxL) 
        
        
        m.optimize_restarts(num_restarts = 5,verbose=False)
        ## now 2D GP  
        #gp.fit(X,Y)

        Y_max = Y_mean + np.max(m.predict_noiseless(X)[0])*Y_std
        #Y_max = Y_mean + np.max(gp.predict(X))*Y_std

        # store gp object and key parameters
        self.gp.append([m,Y_mean,Y_std,Y_max])
        #self.gp.append([gp,Y_mean,Y_std,Y_max])


        all_halton_points = self.halton_generator.random(self.n_points)
        self.skip += self.n_points
        all_halton_points = self.halton_start + self.halton_range*all_halton_points
        plausible_indexes = self.calculate_plausibility(all_halton_points)
        all_halton_points = all_halton_points[plausible_indexes]
        # add new points
        while all_halton_points.shape[0]<self.n_points:
            #new_halton_points = i4_halton_generate(self.input_dim,self.n_points,skip=self.skip)
            #self.skip += self.n_points
            new_halton_points = self.halton_generator.random(1)
            self.skip += 1
            new_halton_points = self.halton_start + self.halton_range*new_halton_points
            plausible_indexes = self.calculate_plausibility(new_halton_points)
            new_halton_points = new_halton_points[plausible_indexes]
            all_halton_points= np.vstack((all_halton_points,new_halton_points))

        self.halton_points = np.vstack((self.halton_points,all_halton_points))
        self.likelihood = np.append(self.likelihood,np.full(all_halton_points.shape[0],np.nan))
        self.sim_output = self.sim_output + [None]*all_halton_points.shape[0]
        return

    def remove_implausible(self):
        # remove implausible points
        plausible_indexes = self.calculate_plausibility(self.halton_points)

        self.halton_points = self.halton_points[plausible_indexes]
        self.likelihood = self.likelihood[plausible_indexes]
        self.sim_output = [item for keep, item in zip(plausible_indexes, self.sim_output) if keep]

    def recalculate_likelihoods(self):
        # use the stored simulation output to recalculate with new
        # regression coefficients

        for i, p in tqdm(enumerate(self.halton_points)):

            # points have a nan likelihood if they haven't been evaluated
            # so we skip points that are finite to include previous sims in the GP
            if np.isnan(self.likelihood[i]):
                continue
            self.likelihood[i] = self.likelihood_function(self.sim_output[i],self.reg_coeff)

    def update_rc(self):
        # update the regression coefficients using latest plausible points
        Y = self.halton_points[np.isfinite(self.likelihood)]
        
        #return if we don't have at least 2 plausible locations
        if len(Y)<=1: 
            return
        X = np.array([item for keep, item in zip(np.isfinite(self.likelihood), self.sim_output) if keep])
        
        likelihood = np.squeeze(self.predict_final(Y)[0])
        
        idx = -likelihood.argsort()[:self.n_points]
        
        #maxid = likelihood.argmax()
        #dist = (Y[:,0]-Y[maxid,0])**2+(Y[:,1]-Y[maxid,1])**2
        #idx = dist.argsort()[:self.n_points]
        
        Y = Y[idx]
        X = X[idx]

        # Y is (npoints x nparams)
        # X is (npoints x nstats x nrepeats)
        # first tile Y and reshape to be (npoints*nrepeats x nparams)
        Y = np.reshape(np.tile(Y[:,None,:],reps=(1,X.shape[-1],1)),(-1,Y.shape[-1]))
        # transpose X to be (npoints x nrepeats x nstats)
        X = np.transpose(X,axes=[0,2,1])
        # reshape X to be (npoints*nrepeats x nstats)
        X = X.reshape((-1,X.shape[-1]))

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(X, Y)
        y_pred = regr.predict(X)
        print('New regression model coefficient of determination: ', r2_score(Y,y_pred,multioutput='raw_values'))
        
        self.reg_coeff = regr.coef_
        self.recalculate_likelihoods()
        return

    def calculate_plausibility(self, X):
        # calculate implausible points

        X = np.atleast_2d(X)
        
        assert X.shape[-1] == self.input_dim, "input dimension mismatch"
        
        plausible = np.ones(np.shape(X)[0],dtype=np.bool)
        #T = 0.5 #2.0
#         for gp_and_params in self.gp:
#             gp, Y_mean, Y_std, Y_max = gp_and_params
#             y_pred, y_pred_std = gp.predict(X,return_std=True)
#             y_pred = Y_mean + (y_pred * Y_std)
#             y_pred_std = y_pred_std * Y_std
#             plausible[(np.squeeze(y_pred) + 3*y_pred_std)<(Y_max-self.T)]=0

#         return plausible
        # gpy
        for gp_and_params in self.gp:
            gp, Y_mean, Y_std, Y_max = gp_and_params
            y_pred, y_pred_cov = gp.predict_noiseless(X)#,return_std=True)
            y_pred = Y_mean + (y_pred * Y_std)
            y_pred_std = np.squeeze(y_pred_cov)**0.5 * Y_std
            plausible[(np.squeeze(y_pred) + 3*y_pred_std)<(Y_max-self.T)]=0

        return plausible

    def predict_final(self, X, remove_implausible=True):
        # predict final wave GP at the input location

        X = np.atleast_2d(X)
        
        assert X.shape[-1] == self.input_dim, "input dimension mismatch"
        
#         gp, Y_mean, Y_std, _ = self.gp[-1]
#         y_pred, y_pred_std = gp.predict(X,return_std=True)
#         y_pred = Y_mean + (y_pred * Y_std)
#         y_pred_std = y_pred_std * Y_std

#         if remove_implausible:
#             plausible_indexes = self.calculate_plausibility(X)
#             y_pred[(plausible_indexes==False)] = -np.inf
#             y_pred_std[(plausible_indexes==False)] = -np.inf

#         return y_pred, y_pred_std

        #gpy
        gp, Y_mean, Y_std, _ = self.gp[-1]
        y_pred, y_pred_cov = gp.predict_noiseless(X)#,return_std=True)
        y_pred = Y_mean + (y_pred * Y_std)
        y_pred_std = (y_pred_cov)**0.5 * Y_std
        
        if remove_implausible:
            plausible_indexes = self.calculate_plausibility(X)
            y_pred[(plausible_indexes==False)] = -np.inf
            y_pred_std[(plausible_indexes==False)] = -np.inf

        return y_pred, y_pred_std



