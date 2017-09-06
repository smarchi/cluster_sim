def cluster_gen(N_star_cluster,n_bkg,L,re,n,x0,y0,bkg='poisson',plot_dist=True,plot_obj=True):

    import numpy as np
    from scipy import special,interpolate
    from matplotlib import pyplot as plt
    from astropy.table import Table

    #This code creates a simulated stellar object using a sersic density profile with 0 eccentricity

    #Parameters
    #N_star_cluster: number of member stars
    #n_bkg: background density
    #L: FOV size
    #re: effective radius
    #n: Sersic index
    #x0: central x coordinate (cartesian)
    #y0: Central y coordinate (cartesian)
    #bkg: Statistics for computing background. Choose from 'poisson' or 'uniform'. The default is 'poisson'.
    #plot_dist: plot density profile. TRUE or FALSE. TRUE is default
    #plot_obj: plot simulated object. TRUE or FALSE. TRUE is default

    #Returns: two astropy Tables. The first contains the member stars coordinates, 
    #the second contains the background stars coordinates.
    

    #Specify some parameters
    b=1.9992*n-0.3271
    alfa=re/(b**n)
    central_den=N_star_cluster/(2.0*np.pi*(alfa**2)*n*special.gamma(2.0*n))
    
    #Sersic density profile function
    def dist(r,central_den,re,n):
    
        b=1.9992*n-0.3271
        alfa=re/(b**n)

        return central_den*np.exp(-1.0*((r/alfa)**(1.0/n)))

    #get values for dist
    dr=0.001
    r=np.arange(0,10*re,dr)
    rho=dist(r,central_den,re,n)

    if plot_dist:
        plt.figure()
        plt.scatter(r,rho,s=2)
        plt.show()

    #create cdf for dist
    area_r=np.pi*(r[1:]**2-r[0:-1]**2)
    cumfunc=np.cumsum(area_r*rho[0:-1])
    cumfunc=cumfunc/np.max(cumfunc)

    r=r[0:-1]
    
    if plot_dist:
        plt.figure()
        plt.scatter(r,cumfunc,s=2)
        plt.show()

    #inverted cdf
    inv_cumfunc=interpolate.interp1d(cumfunc,r,fill_value='extrapolate')

    #random values from cumfunc
    a=np.random.rand(N_star_cluster)

    #random values of r
    r_sim=inv_cumfunc(a)

    #random values of theta
    t=np.random.uniform(0,2.0*np.pi,N_star_cluster)
    x_sim=r_sim*np.cos(t)+x0
    y_sim=r_sim*np.sin(t)+y0

    ## add background
    x_bkg=np.array([])
    y_bkg=np.array([])

    #uniform background
    if bkg=='uniform':
        N_star_bkg=int((L**2)*n_bkg)
        x_bkg=np.random.uniform(0,L,int(N_star_bkg))
        y_bkg=np.random.uniform(0,L,int(N_star_bkg))
    
    #Poissonian background
    elif bkg=='poisson':
        dx=0.5
        dy=0.5
        x_grid=np.linspace(0,L,L/dx)
        y_grid=np.linspace(0,L,L/dy)

        N_stars_bkg=np.zeros((len(x_grid)-1)*(len(y_grid)-1)).reshape(len(y_grid)-1,len(x_grid)-1)
        x_bkg=np.array([])
        y_bkg=np.array([])

        for i in range(len(x_grid)-1):
            for j in range(len(y_grid)-1):
                A=(x_grid[i+1]-x_grid[i])*(y_grid[j+1]-y_grid[j])
                N_star_bkg_aux=A*n_bkg

                #random stars with poisson distribution
                N_stars_bkg[j,i]=int(np.random.poisson(N_star_bkg_aux,1))
                x_bkg=np.append(x_bkg,np.random.uniform(x_grid[i],x_grid[i+1],N_stars_bkg[j,i]))
                y_bkg=np.append(y_bkg,np.random.uniform(y_grid[j],y_grid[j+1],N_stars_bkg[j,i]))

    #plot simulated object
    if plot_obj:
        plt.figure(figsize=(10,10))
        plt.scatter(x_sim,y_sim,s=2,color='b')
        plt.xlim(0,L)
        plt.ylim(0,L)
    
        plt.scatter(x_bkg,y_bkg,s=2,color='b')
        circle1=plt.Circle((x0,y0),re,color='r',fill=False,lw=2)
        plt.gcf().gca().add_artist(circle1)
        plt.show()
    
    #save simulated data in table
    table_sim_cluster=Table([np.arange(len(x_sim)),x_sim,y_sim,r_sim],names=('ID','x','y','r'))
    r_bkg=np.sqrt((x_bkg-x0)**2+(y_bkg-y0)**2)
    table_sim_bkg=Table([x_bkg,y_bkg,r_bkg],names=('x','y','r'))
    
    return table_sim_cluster,table_sim_bkg