from floris_obj_cons import Floris
import os
import numpy as np
import torch
from multiprocessing import Pool
from matplotlib import pyplot as plt

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
no_turbine = 8
file_hf = 'inputs/gch.yaml'
floris_hf = Floris(file_hf, n_turbines=no_turbine, wd_no_bins=18) # for HF we use 18 bins and for LF we use 6 bins

file_lf = 'inputs/jensen.yaml'
floris_lf = Floris(file_lf, n_turbines=no_turbine, wd_no_bins=6)


def parallel_obj_hf(x):
    return floris_hf.evaluate_obj(x)

def parallel_obj_lf(x):
    return floris_lf.evaluate_obj(x)

# initialize the FLORIS interface fi
# TODO : need to add MF here

def AEP_objective_hf(x,parallelize=True):
    """
    Calculate the Annual Energy Production (AEP) objective for the given input using hf solver.

    Parameters:
    - x: list, input design variables. len of x is 2*no_turbines. The first half of x is the x coordinates and the second half is the y coordinates.
    - parallelize: bool, flag indicating whether to parallelize the calculation, using multiprocessing. Adjust the pool according to the number of cores available.

    Returns:
    - aep: numpy array or float, calculated AEP value(s)
    """
    # time the below code
    if len(x.shape) == 2:
        if parallelize:
            with Pool(16) as p:
                aep = p.map(parallel_obj_hf, x)
        # loop over the rows
        else:
            aep = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                #start = time.time()
                aep[i] = floris_hf.evaluate_obj(x[i,:])
                #end = time.time()
                #print(f'Time taken for HF AEP objective is {end - start}')
        #end = time.time()
        #print(f'Time taken for AEP objective is {end - start}')
    else:
        aep = floris_hf.evaluate_obj(x)
    return aep

def AEP_objective_lf(x,parallelize=True):
    """
    Calculate the Annual Energy Production (AEP) objective for the given input using lf solver.

    Parameters:
    - x: list, input design variables. len of x is 2*no_turbines. The first half of x is the x coordinates and the second half is the y coordinates.
    - parallelize: bool, flag indicating whether to parallelize the calculation, using multiprocessing. Adjust the pool according to the number of cores available.

    Returns:
    - aep: numpy array or float, calculated AEP value(s)
    """
    if len(x.shape) == 2:
        if parallelize:
            with Pool(32) as p:
                aep = p.map(parallel_obj_lf, x)
        # loop over the rows
        else:
            aep = np.zeros(x.shape[0])

            for i in range(x.shape[0]):
                #start = time.time()
                aep[i] = floris_lf.evaluate_obj(x[i,:])
                #end = time.time()
                #print(f'Time taken for LF AEP objective is {end - start}')
    else:
        aep = floris_lf.evaluate_obj(x)
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
    scout =True

    dim =2*no_turbine

    # 16 dim
    # (32,8)
    # 48 dim
    # (96,24)


    #x0_mean = get_init() # 24 turbines
    x0_mean = [0.2,0.5,0.8,0.2,0.5,0.8,0.33,0.66,0.2,0.2,0.2,0.5,0.5,0.5,0.8,0.8] # 8 turbines
    # x and y coordinates for 16 turbines in a grid in a unit square
    #x0_mean = [0.1,0.3,0.5,0.1,0.3,0.5,0.1,0.3,0.5,0.1,0.1,0.1,0.3,0.3,0.3,0.5,0.5,0.5]
    x0_std = np.log(0.1*np.ones(len(x0_mean)))  # exp of e is expected.
    # stack the two together in a new list 
    x0 = np.hstack((x0_mean,x0_std.tolist())).tolist()

    # time it
    start = time.time()
    obj = MultifidelityObjective(dim=dim,f_list=[AEP_objective_lf,AEP_objective_hf],constraints= [space_constraint_hf,distance_from_boundaries_constraint_hf], qmc=True)
    #obj.set_num_samples([128, 32])
    obj.set_num_samples([32, 8])
    optimizer = Stochastic_Optimizer(obj,initial_val = x0,natural_gradients=True, verbose=True,tol_constraints =1e-01,tolerance_sigma = 5e-04,tolerance_theta=1e-05)
    #lr = 1e-2
    lr = 5e-2
    optimizer.create_optimizer('Adam', lr=lr)
    optimizer.optimize(num_lambdas=25, num_steps_per_lambda=300)

    end = time.time()

    total_time = end - start
    print(f'Time taken for the MF optimization is {total_time}')

    #obj = Baseline1(dim=dim,func=AEP_objective_hf,constraints=[space_constraint_hf,distance_from_boundaries_constraint_hf], num_samples=128)
    #optimizer = Stochastic_Optimizer(obj,initial_val = x0,natural_gradients=True, verbose=True,tol_constraints =1e-01,tolerance_sigma = 5e-05,tolerance_theta=1e-05)

    

    results_obj_cons = optimizer.get_objective_constraint_evolution() 
    results_x = optimizer.get_design_variable_evolution()
    plot_path = os.getcwd() + f'/Results/Plots/MF/lr_{lr}'
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
    # floris_viz.wind_contour(floris_hf.fi, des_unnorm_init, wind_direction=270.0, title='Initial Layout')
    # plt.savefig(plot_path + f'/scout_windwake_init_{date}.pdf')

    # --- optimized windwake
    floris_viz.wind_contour(floris_hf.fi, des_unnorm_opt,wind_direction=360,clevels=25,color_bar=True, title=f'Optimal layout \n(Wind Direction $360^o$, Wind speed $8m/s$)')
    plt.savefig(plot_path + f'/scout_windwake_opt_{date}.pdf',bbox_inches='tight')


    # save the results
    result_path = os.getcwd() + f'/Results/MF/lr_{lr}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    optimizer.save_results(result_path)
    # save the AEP values in .json for both scipy and scout
    AEP_values = {'AEP_init': AEP_init, 'AEP_final_scout': AEP_final, 'total_time': total_time}
    with open(result_path + f'/AEP_values_MF_{date}.json', 'w') as f:
        json.dump(AEP_values, f)







