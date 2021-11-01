# implementation of Haario, Heikki; Saksman, Eero; Tamminen, Johanna. An adaptive Metropolis algorithm. Bernoulli 7 (2001), no. 2, 223--242. 
# based on code from https://github.com/philipk01/Optimization_and_Sampling_for_Bayesian_Inference

import numpy as np
from tqdm import tqdm

def update_moments(mean, M2, sample, n):
    next_n = n + 1
    w = 1/next_n
    new_mean = mean + w*(sample - mean)
    delta_bf, delta_af = sample - mean, sample - new_mean
    new_M2 = M2 + np.outer(delta_bf, delta_af)
    return new_mean, new_M2

def get_proposal_cov(M2, n, C_0, beta=0.05):
    d, _ = M2.shape
    init_period = 2*d
    s_opt = 2.38/np.sqrt(d)
    if np.random.rand()<=beta or n<= init_period:
        return C_0
    else:
        # We can always divide M2 by n-1 since n > init_period
        return (s_opt/(n - 1))*M2



def generate_AM_candidate(current, M2, n, cov_0):
    prop_cov = get_proposal_cov(M2, n, cov_0)
    current = np.atleast_1d(current)
    try:
        candidate = np.random.multivariate_normal(mean=current,cov=prop_cov,check_valid='raise')
    except:    
        candidate = np.random.multivariate_normal(mean=current, cov=cov_0)
    
    return candidate


def am_sampler(likelihood_function, ndim, init_p, prior, sigma, n_samples, burn_in, m): 
    cov_0 = np.diag(sigma)#0.1/np.sqrt(ndim)
    #idty = np.eye(ndim)
    #cov_0 = sigma_0**2*idty
    #cov_opt = sigma**2*idty
    
    mean = np.zeros(ndim)
    M2 = np.zeros((ndim,ndim))
    # array for samples
    samples = np.zeros((n_samples,ndim))
        
    # store initial value
    #samples[0] = init_p.squeeze()

    # reshape and compute the current likelihood
    z = np.squeeze(init_p)#,(ndim))
    
    l_cur, std_cur  = likelihood_function(z)
    l_cur = np.squeeze(l_cur)
     
    
    # check is finite and we're starting in plausible region
    if not np.isfinite(l_cur):
        print('implausible starting location')
        return
        
    # total iterations for number of samples
    iters = (n_samples * m) + burn_in

    # create random numbers outside the for loop
    #innov = np.random.normal(loc=0, scale=sigma, size=(iters,ndim))
    u = np.random.rand(iters) 
    #for i in tqdm(range(iters)): 
    for i in range(iters): 
        # new location for z
        #cand = z + innov[i]
        cand = generate_AM_candidate(z,M2,i,cov_0)

    
        l_cand, std_cand  = likelihood_function(np.squeeze(cand))
        l_cand = np.squeeze(l_cand)  
        
        if ndim==1:
            if cand<prior[0]:
                l_cand = - np.inf
            elif cand>prior[1]:
                l_cand = - np.inf
        else:        
            for q in range(ndim):
                if cand[q]<prior[q,0]:
                    l_cand = - np.inf
                elif cand[q]>prior[q,1]:
                    l_cand = - np.inf                           
                                   
                             

        # check not moving to implausible location
        if np.isfinite(l_cand):
            # use sample from GP for update to account for uncertainty
            l_cand_s = np.random.normal(loc=l_cand, scale=std_cand)
            l_cur_s = np.random.normal(loc=l_cur, scale=std_cur)
            # Accept or reject candidate
            if np.exp(l_cand_s - l_cur_s) > u[i]:
                z = cand
                l_cur = l_cand
                std_cur = std_cand     

        mean, M2 = update_moments(mean, M2, z, i)

        # Only keep iterations after burn-in and for every m-th iteration
        if i >= burn_in and i % m == 0:
            samples[(i - burn_in) // m] = z.squeeze() 

           

    return samples

