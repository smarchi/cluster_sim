def mcmc_density_fit(init_guess,args,model):

    import emcee
    import numpy as np
    import density_profile_models as dpm

    #Performs a mcmc fit to simulated cluster.
    #init_guess: initial guess for parameters
    #args: extra arguments
    #model: 'sersic' or 'exp'
    
    def lnprior_exp(pars,bkg_lim):
        ra0,dec0,ell,re,n_bkg=pars
        if 0 < ra0 < 60.0 and 0 < dec0 < 60.0 and 0 <= ell < 1.0 and 0.1 <= re < 40  and\
        0 < n_bkg < bkg_lim:
            return 0.0
        return -np.inf
    
    def lnprior_sersic(pars):
        ra0,dec0,ell,re,n=pars
        if 0 <= ra0 < 60 and 0 <= dec0 < 60 and 0 <= ell < 1.0 and 0.01 < re < 30 and\
        0.2 < n < 20.0:
            return 0.0
        return -np.inf
    
    def lnprob_exp(pars,ra_i,dec_i,Ntot,A,bkg_lim):
        lp=lnprior_exp(pars,bkg_lim)
        if not np.isfinite(lp):
            return -np.inf
        return lp+dpm.exp_profile(pars,ra_i,dec_i,Ntot,A)*-1.0
    
    def lnprob_sersic(pars,ra_i,dec_i,Area,Ntot,n_bkg):
        lp=lnprior_sersic(pars)
        if not np.isfinite(lp):
            return -np.inf
        return lp+dpm.sersic_profile(pars,ra_i,dec_i,Area,Ntot,n_bkg)*-1.0
    
    ndim, nwalkers,steps =len(init_guess),100,500
    pos = [init_guess+1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    if model=='exp':
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_exp, args=args)   
    elif model=='sersic':
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_sersic, args=args)
    
    sampler.run_mcmc(pos, steps)
    
    return sampler

def walkers_plot(sampler,nrows,ncols,plot_number,figsize,titles):

    #Plot a grid with chains from the different parameters
    #Parameters:
    #sampler: sampler from mcmc fit. Emcee object
    #nrows: number of rows for subplot grid. Integer.
    #ncols: number of columns for subplot grid. Integer.
    #plot_number: total number of subplots. Integer.
    #figsize: size of grid. Tuple of integers.
    #titles: array of title names. Array of strings.

    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.reset_orig()
    plt.figure(figsize=figsize)
    for i in range(plot_number):
        for j in range(100):
            plt.subplot(nrows,ncols,i+1)
            plt.plot(sampler.chain[j,:,i],color='black')
            plt.title(titles[i])
    plt.show()