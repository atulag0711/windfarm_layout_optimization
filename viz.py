import floris.tools.visualization as wakeviz
import floris
from floris.tools import FlorisInterface
from floris.tools.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)
import floris.tools as wfct
import numpy as np
import os
import torch
import pickle
from matplotlib import pyplot as plt
from matplotlib import rc
from itertools import product
from typing import Union

from scoutNd.stochastic_optimizer import Stochastic_Optimizer
from scoutNd.objective_function import Baseline1
# add bm, amsmath and all that to matplotlib


rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#params= {'text.latex.preamble' : r'\usepackage{amsmath,bm, amsfonts}'}
#plt.rcParams.update(params)
rc('text.latex', preamble=r'\usepackage{amsmath,bm,amsfonts}')
# plt.style.use('ggplot')
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 20

rc('font', size=MEDIUM_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure titles

torch.set_default_dtype(torch.float64)
#fix the seed
# seed = 666
# np.random.seed(seed)
# torch.random.manual_seed(seed)
import time
datetime = time.strftime("%Y%m%d-%H%M%S")

class floris_viz:
    @staticmethod
    def plot_layout_optimization(init_des:Union[np.ndarray,list], final_des:Union[np.ndarray,list]):
        """
        Plot the layout optimization results.

        Parameters:
        init_des (Union[np.ndarray, list]): Initial design coordinates.
        final_des (Union[np.ndarray, list]): Final optimized design coordinates.
        """
        # if arguments are list, convert to numpy array
        if isinstance(init_des, list):
            init_des = np.array(init_des)
        if isinstance(final_des, list):
            final_des = np.array(final_des)

        # x is first half of the init_des vector
        x_initial = init_des[:len(init_des)//2]
        # y is second half of the init_des vector
        y_initial = init_des[len(init_des)//2:]

        # x is first half of the final_des vector
        x_opt = final_des[:len(final_des)//2]
        # y is second half of the final_des vector
        y_opt = final_des[len(final_des)//2:]

        plt.figure(figsize=(6, 4))
        fontsize = 18
        plt.plot(x_initial, y_initial, "ob")
        plt.plot(x_opt, y_opt, "or")
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel(r"$x/x_{max}$", fontsize=fontsize)
        plt.ylabel(r"$y/y_{max}$", fontsize=fontsize)
        plt.axis("equal")
        plt.grid()
        plt.tick_params(which="both", labelsize=fontsize)
        plt.legend(
            ["Old locations", "New locations"],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=fontsize,
        )
        # make a layout box with the boundary wind farm with blue line. its a square with length 1
        verts = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        for i in range(len(verts)):
            if i == len(verts) - 1:
                plt.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "b")
            else:
                plt.plot(
                    [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "b"
                )

    @staticmethod
    def wind_contour(fi:FlorisInterface, layout:list, wind_direction:float=270.0, title:str='Wind Contour', **kwargs):
        """
        Plot wind contour for a given layout and wind direction.

        Parameters:
        fi (FlorisInterface): The FlorisInterface object.
        layout (list): The layout of wind turbines. Expects unnomralized layout.
        wind_direction (float, optional): The wind direction in degrees. Default is 315.0.

        Returns:
        im (matplotlib.image.AxesImage): The image of the wind contour plot.
        """
        # reinitialize the floris interface with the new layout
        # TODO: the plot seems to be tiled, cant get it to be straight.
        fi.reinitialize(layout_x=layout[:len(layout)//2], layout_y=layout[len(layout)//2:])

        # if x_bounds and y_bounds in kwargs, use them, else use the default values. also pop them out.
        if 'x_bounds' in kwargs:
            x_bounds = kwargs.pop('x_bounds')
        else:
            x_bounds = None
        if 'y_bounds' in kwargs:
            y_bounds = kwargs.pop('y_bounds')
        else:  
            y_bounds = None

        horizontal_plane = fi.calculate_horizontal_plane(
            height=90.0,
            wd=[wind_direction],
            x_bounds= x_bounds,
            y_bounds= y_bounds
        )


        # Create the plots
        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        #ax_list = ax_list.flatten()
        im = wakeviz.visualize_cut_plane(
            horizontal_plane,
            ax=ax_list,
            title=title,
            **kwargs
        )
        return im
    @staticmethod
    def power_output_comparision(x_mean_hf:list, x_mean_lf:list, savefig=None):
        """
        Compare the power output for different layouts.

        Parameters:
        - x_mean_hf (list): List of x-coordinates for the high-fidelity layout.
        - x_mean_lf (list): List of x-coordinates for the low-fidelity layout.
        - savefig (str, optional): File path to save the generated plots. Defaults to None.
        """
        
        # get init layout, not general yet, but can be adopted
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
        x_init = get_init()

        # unnormaize
        x_mean_mf = x_mean_mf*333.33*len(x_mean_mf)//2
        x_mean_hf = x_mean_hf*333.33*len(x_mean_hf)//2
        x_init = np.array(x_init)*333.33*len(x_init)//2

        def get_power_for_given_layout(x):
            degress = np.linspace(0,360,72)
            # aray of size degrees
            power = np.ndarray((len(degress),))
            for i in range(len(degress)):
                fi = wfct.floris_interface.FlorisInterface("inputs/gch.yaml")
                fi.reinitialize(layout_x=x[:len(x)//2], layout_y=x[len(x)//2:])
                fi.reinitialize(wind_directions=[degress[i]],wind_speeds=[8])
                fi.calculate_wake()
                power[i] = fi.get_farm_power()/1e06
            # flatten the list
            return power # in MW

        # get power for the init, hf and mf
        power_init = get_power_for_given_layout(x_init)
        power_hf = get_power_for_given_layout(x_mean_hf)
        power_mf = get_power_for_given_layout(x_mean_mf)
        
        # plot the results in two subplots. first: with x axis degress and y with power. second: a radialt plot with the power
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        degress = np.linspace(0, 360, len(power_init))
        # arrange 360 in 18 bins
        ax.plot(degress, power_init, label='Initial grid layout', color='black', linestyle='--')
        ax.plot(degress, power_hf, label='Optimal layout, Scout', color='red')
        ax.plot(degress, power_mf, label='Optimal layout, (MF)Scout',color='blue')
        ax.set_xlabel('Wind direction ($^o$)')
        ax.set_ylabel('Power (MW)')
        #ax.set_title('Power output for different layouts')
        ax.legend()
        if savefig is None:
            plt.show()
        else:
            plt.savefig(savefig, dpi=300, bbox_inches='tight') # savefig= 'Results/Plots/final_dim_48/power_output_layout_comparision.pdf'

        # radial plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        theta = np.linspace(0, 2*np.pi, len(power_init))
        ax.plot(theta, power_init, label='Initial grid layout', linestyle='--')
        ax.plot(theta, power_hf, label='Optimal layout Scout',color='red')
        ax.plot(theta, power_mf, label='Optimal layout (MF)Scout',color='blue')
        ax.set_rlabel_position(180)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title('Power output for different layouts')
        ax.legend()
        if savefig is None:
            plt.show()
        else:
            plt.savefig(savefig, dpi=300, bbox_inches='tight') # savefig= 'Results/Plots/final_dim_48/power_output_layout_comparision_radial.pdf'