def sersic_profile(pars,ra_i,dec_i,A,Ntot,n_bkg):

    import numpy as np
    from scipy import special

    '''
    Ntot=total number of stars used to calculate parameters
    A=area covered by stars
    ra_i,dec_i=coordinates of stars
    pars=parameters to be determined. ra0,dec0=centroid, e=ellipticity, theta=position angle respect to north,
                                      Re=effective radius, n=sersic index, n_bkg=background density
    '''
    ra0=pars[0]
    dec0=pars[1]
    ell=pars[2]
    theta=0.
    re=pars[3]
    n=pars[4]
     
    b=1.9992*n-0.3271
    alfa=re/(b**n)
    central_density=(Ntot-A*n_bkg)/(2.0*np.pi*(alfa**2)*n*special.gamma(2.0*n)*(1.0-ell))
    
    x=(ra_i-ra0)*np.cos(dec0/60.*np.pi/180.0)
    y=dec_i-dec0
    termr1=(1.0/(1.0-ell))*(x*np.cos(theta)-y*np.sin(theta))
    termr2=x*np.sin(theta)+y*np.cos(theta)
    
    r=np.sqrt(termr1**2+termr2**2)
    
    p=central_density*np.exp(-1.0*((r/alfa)**(1.0/n)))+n_bkg
    
    return -1.0*np.sum(np.log(p))

def exp_profile(pars,ra_i,dec_i,A,Ntot):

    import numpy as np
    from scipy import special

    '''
    Ntot=total number of stars used to calculate parameters
    A=area covered by stars
    ra_i,dec_i=coordinates of stars
    pars=parameters to be determined. ra0,dec0=centroid, e=ellipticity, theta=position angle respect to north,
                                      re=scale radius, n_bkg=background density
    '''
    
    ra0=pars[0]
    dec0=pars[1]
    ell=pars[2]
    theta=0
    re=pars[3]
    n_bkg=pars[4]
    
    n=1.0
    b=1.9992*n-0.3271
    alfa=re/(b**n)
    
    central_density=(Ntot-A*n_bkg)/(2.0*np.pi*(alfa**2)*n*special.gamma(2.0*n)*(1.0-ell))
    x=(ra_i-ra0)*np.cos(dec0*np.pi/180.0)
    y=dec_i-dec0
    
    termr1=(1.0/(1.0-ell))*(x*np.cos(theta)-y*np.sin(theta))
    termr2=x*np.sin(theta)+y*np.cos(theta)
    
    r=np.sqrt(termr1**2+termr2**2)*60.0
    
    p=central_density*np.exp(-1.68*r/re)+n_bkg
    
    return -1.0*np.sum(np.log10(p))