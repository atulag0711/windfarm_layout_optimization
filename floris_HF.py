from floris_obj_cons import Floris
import os
import numpy as np
import torch
from multiprocessing import Pool
from matplotlib import pyplot as plt
from pyDOE import lhs

# optimization modules
from scoutNd.stochastic_optimizer import Stochastic_Optimizer
from scoutNd.multifidelity_objective import MultifidelityObjective
from scoutNd.objective_function import *
from scoutNd.viz import variable_evolution
from viz import floris_viz  
# set float 64 as default
torch.set_default_dtype(torch.float64)
import time, json
date = time.strftime("%Y%m%d-%H%M%S")

# import scout modules

file_dir = os.path.dirname(os.path.abspath(__file__))
no_turbine = 24
file_hf = 'inputs/gch.yaml'
floris_hf = Floris(file_hf, n_turbines=no_turbine, wd_no_bins=18) # for HF we use 18 bins and for LF we use 6 bins


def parallel_obj_hf(x):
    return floris_hf.evaluate_obj(x)


def AEP_objective_hf(x, parallelize=True):
    """
    Calculate the Annual Energy Production (AEP) objective for the given input.

    Parameters:
    - x: list, input design variables. len of x is 2*no_turbines. The first half of x is the x coordinates and the second half is the y coordinates.
    - parallelize: bool, flag indicating whether to parallelize the calculation, using multiprocessing. Adjust the pool according to the number of cores available.

    Returns:
    - aep: numpy array or float, calculated AEP value(s)
    """

    # time the below code
    start = time.time()
    if len(x.shape) == 2:
        if parallelize:
            with Pool(32) as p:
                aep = p.map(parallel_obj_hf, x)
        else:
            aep = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                aep[i] = floris_hf.evaluate_obj(x[i,:])
    else:
        aep = floris_hf.evaluate_obj(x)
    return aep

def space_constraint_hf(x):
    """
    Evaluate the space constraint for the given input.

    Parameters:
    - x: list, input design variables. len of x is 2*no_turbines. The first half of x is the x coordinates and the second half is the y coordinates.

    Returns:
    numpy.ndarray: Array of space constraint values.
    """

    #TODO: think about the c2
    if len(x.shape) == 2:
        # loop over the rows
        c1 = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            c1[i] = floris_hf.evaluate_space_constr(x[i,:])
            # if any compnent of c1 is positve, print it and the correspomndiong x row
            # if c1[i]>0:
            #     print(f'c1 is {c1[i]} and x is {x[i,:]}')
    else:
        c1 = floris_hf.evaluate_space_constr(x)
        c1 =np.array([c1])
    #c1, c2 = floris_hf.evaluate_constr(x)
    return c1

def distance_from_boundaries_constraint_hf(x):
    """
    Calculates the distance from boundaries constraint for the given input.

    Parameters:
    - x: list, input design variables. len of x is 2*no_turbines. The first half of x is the x coordinates and the second half is the y coordinates.

    Returns:
    numpy.ndarray: Array of shape (n,) containing the distance from boundaries constraint values.
    """

    if len(x.shape) == 2:
        # loop over the rows
        c2 = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            c2[i] = floris_hf.evaluate_distance_from_boundries_constr(x[i,:])
    else:
        c2 = floris_hf.evaluate_distance_from_boundries_constr(x)
        c2 = np.array([c2])
    return c2


def get_init():
    # Number of points
   # Number of points
    n_points = 24

    # Determine grid structure
    n_rows = 6
    n_cols = 4

    # New square boundaries
    x_min, y_min = 0.1, 0.1
    x_max, y_max = 0.9, 0.9

    # Calculate spacing between points
    x_spacing = (x_max - x_min) / (n_cols - 1)
    y_spacing = (y_max - y_min) / (n_rows - 1)

    # Generate coordinates within the new boundaries
    x_coordinates = [x_min + i * x_spacing for i in range(n_cols) for _ in range(n_rows)]
    y_coordinates = [y_min + j * y_spacing for _ in range(n_cols) for j in range(n_rows)]

    # Combine into a list
    coordinates = x_coordinates + y_coordinates
    # Print or return coordinates
    print(coordinates)
    return coordinates



if __name__ == '__main__':

    # testing optimization
    scout =False
    scipy =True

    dim =2*no_turbine

    #x0_mean = get_init() # [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3666666666666667, 0.3666666666666667, 0.3666666666666667, 0.3666666666666667, 0.3666666666666667, 0.3666666666666667, 0.6333333333333333, 0.6333333333333333, ...]
    # cretae a list of of length 6 with values randomble gemnerate between 0-1
    #x0_mean = np.random.rand(dim).tolist()
    # TODO: start both scipy and scout with the same initial uniform layout. Do for 8 tubines. Then use scount for 20 turbines. Do MF with 8 turbines.
    #x0_mean = [0.3,0.35,0.75,0.3,0.35,0.25] 
    #x0_mean = [0.2,0.5,0.8,0.2,0.5,0.8,0.33,0.66,0.2,0.2,0.2,0.5,0.5,0.5,0.8,0.8] # 8 turbines
    #x0_mean = [0.1,0.3,0.5,0.1,0.3,0.5,0.1,0.3,0.5,0.1,0.1,0.1,0.3,0.3,0.3,0.5,0.5,0.5]

    #latin hypercube sample
    x0_mean = lhs(dim, samples=1).tolist()[0]
    x0_std = np.log(0.1*np.ones(len(x0_mean)))  # exp of e is expected.
    # stack the two together in a new list 
    x0 = np.hstack((x0_mean,x0_std.tolist())).tolist()
    

    if scout:
        # no of turbines = 8 setup
        samples = 32 #124 in th experiments
        tol_sigma = 1e-04   #   1e-04
        # no of turbines = 24 setup
        #samples = 256
        #tol_sigma = 5e-04

        obj = Baseline1(dim=dim,func=AEP_objective_hf,constraints=[space_constraint_hf,distance_from_boundaries_constraint_hf], num_samples=samples) # used 128 samples for 8 turbines
        optimizer = Stochastic_Optimizer(obj,initial_val = x0,natural_gradients=True, verbose=True,tol_constraints =1e-01,tolerance_sigma = tol_sigma,tolerance_theta=1e-05)
        lr = 5e-2 # 5e-01 and smaller works. 
        optimizer.create_optimizer('Adam', lr=lr)
        optimizer.optimize(num_lambdas=20, num_steps_per_lambda=300)

        results_obj_cons = optimizer.get_objective_constraint_evolution() 
        results_x = optimizer.get_design_variable_evolution()
        plot_path = os.getcwd() + f'/Results/Plots/lr_{lr}'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        
        optimizer.plot_results(plot_path, f'floris_windfarm_{dim}')
        # plot the optimization results
        # evo_plot = variable_evolution(L_x=results_obj_cons[0],f_x=results_obj_cons[1],C_x=results_obj_cons[2],mu = results_x[0], beta=results_x[1],path=plot_path,save_name='floris')
        # evo_plot.plot_all()
        print(f"The optimization is completed in {optimizer.iteration} iterations")
        # results
        #print(f'The final layout is {optimizer.get_final_state['mean']} and the final std is {optimizer.get_final_state['variance']}')
        print(f'The final state is {optimizer.get_final_state()}')

        AEP_init = floris_hf.get_AEP(x0_mean)
        AEP_final = floris_hf.get_AEP(optimizer.get_final_state()['mean'].tolist())
        percent_gain = 100 * (AEP_final - AEP_init) / AEP_init

        print(f'The AEP for the initial layout is {AEP_init} MWh and the AEP for the final layout is {AEP_final} MWh. Percent gain is {percent_gain}')

        

        # plot layout 
        des_unnorm_init, des_unnorm_opt = floris_hf.get_unnorm_initial_final_values(x_init=x0_mean,x_opt=optimizer.get_final_state()['mean'].tolist())
        floris_viz.plot_layout_optimization(init_des=x0_mean, final_des=optimizer.get_final_state()['mean'].tolist())
        #floris_viz.plot_layout_optimization(init_des=des_unnorm_init, final_des=des_unnorm_opt)
        plt.savefig(plot_path + f'/scout_layout_optimization_{date}.pdf')

        # wind wake plots
        # --- init windwake
        floris_viz.wind_contour(floris_hf.fi, des_unnorm_init, wind_direction=360,clevels=25,color_bar=True, title=f'Initial layout \n(Wind Direction $360^o$, Wind speed $8m/s$)')
        plt.savefig(plot_path + f'/scout_windwake_init_{date}.pdf',bbox_inches='tight')

        # --- optimized windwake
        floris_viz.wind_contour(floris_hf.fi, des_unnorm_opt, wind_direction=360,clevels=25,color_bar=True, title=f'Optimal layout \n(Wind Direction $360^o$, Wind speed $8m/s$)')
        plt.savefig(plot_path + f'/scout_windwake_opt_{date}.pdf',bbox_inches='tight')


        # save the results
        result_path = os.getcwd() + '/Results'
        optimizer.save_results(result_path)

    # testing scipy 
    
    if scipy:
        #x0_mean = [0.06823551003081459, 0.08379214218639798, 0.10093456127004205, 0.14337951612088318, 0.12006431867282374, 0.09505882986497355, 0.30508351477802376, 0.3365389461732483, 0.39374923679435314, 0.42470964603889777, 0.48449511337447715, 0.35586681365149886, 0.5414245069031843, 0.625479062032294, 0.7455227183856112, 0.7193610168580479, 0.6970176623978089, 0.6025955818607113, 0.9027907185170136, 0.9469388850511565, 0.9470779516655916, 0.9470779510347239, 0.8990531744570587, 0.9469706121317801, 0.09965147630054887, 0.18059155603510754, 0.3425773247392172, 0.4329006621229973, 0.6032327929034808, 0.9157174765128373, 0.13109348107882562, 0.20604140992480854, 0.37020120258839884, 0.4995208606217588, 0.6810994845418965, 0.8159084140728899, 0.0762086030410173, 0.2201046519869464, 0.31470710738057084, 0.5571861175992768, 0.6406604578886688, 0.8055194888948938, 0.061527937789582585, 0.2382104438279174, 0.3407110636968939, 0.5043825268474327, 0.6935826706939234, 0.8704208122218776]
        floris_scipy_hf = Floris(file_hf, n_turbines=no_turbine, x_init=x0_mean,scipy_opt={"maxiter": 400, "ftol": 1e-14})
        sol = floris_scipy_hf.scipy_optimize()
        
        AEP_scipy = floris_scipy_hf.get_AEP(np.array(sol).flatten())

        # plot layout opt
        plot_path = os.getcwd() + '/Results/Plots'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        floris_scipy_hf.lo.plot_layout_opt_results()
        plt.savefig(plot_path + f'/scipy_layout_optimization_scipy_{date}_{np.mean(x0_mean)}.pdf')

        print(f"The initial layout is {x0_mean} and the initial beta of sigma^2 = e^beta is {x0_std}")

        print(f'The optimized layout is {sol} with scipy')
        print(f'The AEP for the scipy optimized layout is {AEP_scipy} MWh')

        # save the AEP values in .json for both scipy and scout
        # check if AEP_final exists
        AEP_init = floris_hf.get_AEP(x0_mean)
        result_path = os.getcwd() + '/Results'
        #AEP_values = {'AEP_init': AEP_init, 'AEP_final_scout': AEP_final, 'AEP_final_scipy': AEP_scipy}
        if 'AEP_final_scout' in locals():
            AEP_values = {'AEP_init': AEP_init, 'AEP_final_scout': AEP_final, 'AEP_final_scipy': AEP_scipy}
        else:
            AEP_values = {'AEP_init': AEP_init, 'AEP_final_scipy': AEP_scipy}
        with open(result_path + f'/AEP_values_HF_dim_{dim}_{date}.json', 'w') as f:
            json.dump(AEP_values, f)







