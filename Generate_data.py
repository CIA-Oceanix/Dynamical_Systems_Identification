import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
from Dynamical_systems import Lorenz_63, Lorenz_96, oregonator, Adv_Dif_1D
def generate_data(GD):
    """ Generate the true state, noisy observations and catalog of numerical simulations. """

    # initialization
    class xt:
        values = [];
        time = [];
    class catalog:
        analogs = [];
        successors = [];
        source = [];
    
    # use this to generate the same data for different simulations
    np.random.seed(1);
    
    if (GD.model == 'Lorenz_63'):
    
        # 5 time steps (to be in the attractor space)  
        x0 = np.array([8.0,0.0,30.0]);
        S = odeint(Lorenz_63,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        x0 = S[S.shape[0]-1,:];

        # generate true state (xt)
        S = odeint(Lorenz_63,x0,np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        T_test = S.shape[0];      
        t_xt = np.arange(0,T_test,1);       
        xt.time = t_xt*GD.dt_integration;
        xt.values = S[t_xt,:];

        #generate catalog
        S =  odeint(Lorenz_63,S[S.shape[0]-1,:],np.arange(0.01,GD.nb_loop_train+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        T_train = S.shape[0];
        eta = np.random.multivariate_normal(np.zeros(3),GD.sigma2_catalog*np.eye(3,3),T_train);
        catalog_tmp = S+eta;
        catalog.analogs = catalog_tmp[0:-1,:];
        catalog.successors = catalog_tmp[1:,:]
        catalog.source = GD.parameters;
    
    elif (GD.model == 'Lorenz_96'):
        
        # 5 time steps (to be in the attractor space)
        x0 = GD.parameters.F*np.ones(GD.parameters.J);
        x0[np.int(np.around(GD.parameters.J/2))] = x0[np.int(np.around(GD.parameters.J/2))] + 0.01;
        S = odeint(Lorenz_96,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));
        x0 = S[S.shape[0]-1,:];
       

        # generate true state (xt)
        S = odeint(Lorenz_96,x0,np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));       
        T_test = S.shape[0];     
        t_xt = np.arange(0,T_test,1);
        xt.time = t_xt*GD.dt_integration;
        xt.values = S[t_xt,:];

        
        # generate catalog
        S =  odeint(Lorenz_96,S[S.shape[0]-1,:],np.arange(0.01,GD.nb_loop_train+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));        
        T_train = S.shape[0];
        eta = np.random.multivariate_normal(np.zeros(GD.parameters.J),GD.sigma2_catalog*np.eye(GD.parameters.J,GD.parameters.J),T_train);
        catalog_tmp = S+eta;
        catalog.analogs = catalog_tmp[0:-1,:];
        catalog.successors = catalog_tmp[1:,:]
        catalog.source = GD.parameters;
    elif (GD.model == 'oregonator'):
    
        # 5 time steps (to be in the attractor space)       
        x0 = np.array([4,1.1,4]);
        S = odeint(oregonator,x0,np.arange(0,10000+0.000001,GD.dt_integration),args=(GD.parameters.alpha,GD.parameters.beta,GD.parameters.sigma));
        x0 = S[S.shape[0]-1,:];

        # generate true state (xt)
        S = odeint(oregonator,x0,np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration),args=(GD.parameters.alpha,GD.parameters.beta,GD.parameters.sigma));
        T_test = S.shape[0];      
        t_xt = np.arange(0,T_test,1);       
        xt.time = t_xt*GD.dt_integration;
        xt.values = S[t_xt,:];
      

        #generate catalog
        S =  odeint(oregonator,S[S.shape[0]-1,:],np.arange(0.01,GD.nb_loop_train+0.000001,GD.dt_integration),args=(GD.parameters.alpha,GD.parameters.beta,GD.parameters.sigma));
        T_train = S.shape[0];
        eta = np.random.multivariate_normal(np.zeros(3),GD.sigma2_catalog*np.eye(3,3),T_train);
        catalog_tmp = S+eta;
        catalog.analogs = catalog_tmp[0:-1,:];
        catalog.successors = catalog_tmp[1:,:]
        catalog.source = GD.parameters;    
    elif (GD.model == 'Adv_Dif_1D'):
        class catalog:
            num_integration = [];
            true_solution = [];
            euler_integration = []; 
            time = [];
        # 5 time steps (to be in the attractor space)       
        x0 = np.array([GD.parameters.x0]);
        t0 = np.array([GD.parameters.t0]);
        t1 = np.array([GD.nb_loop_train]);
        # true solution
        t = np.arange(0,t1+0.000001,GD.dt_integration)
        true_sol = []
        for i in range(len(t)):
            true_sol.append(x0*np.exp(GD.parameters.w*t0)*np.exp(GD.parameters.w*t[i]))
        euler_sol = [x0]
        for i in range(1,len(t)):
            euler_sol.append(euler_sol[-1]+GD.dt_integration*GD.parameters.w*euler_sol[-1])
                
        r = ode(Adv_Dif_1D).set_integrator('zvode', method='bdf')
        r.set_initial_value(np.reshape(x0,(1,1)), t0).set_f_params(GD.parameters.w)
        t1 = GD.nb_loop_train
        dt = GD.dt_integration
        S = [] 
        while r.successful() and r.t < t1:
            r.integrate(r.t+dt)  
            catalog.num_integration.append(r.y)
            catalog.time.append(r.t)
        catalog.num_integration = np.reshape(np.array(catalog.num_integration)[:,0,0],(len(catalog.num_integration),1))    
        catalog.true_solution = np.array(true_sol)
        catalog.euler_integration = np.array(euler_sol)
    # reinitialize random generator number
    np.random.seed()
    return catalog, xt;