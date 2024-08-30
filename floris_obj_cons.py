from typing import Any
import numpy as np
import floris.tools as wfct
import os
from floris.tools.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)

# https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/windwake.py
# https://github.com/NREL/floris/blob/59e53a66aef134a3c9e912f9468ca667b599d4e5/floris/tools/optimization/legacy/scipy/layout.py#L100

# MF in floris
#  - https://wes.copernicus.org/articles/7/991/2022/  (code : https://zenodo.org/records/6109699)
# Initialize the FLORIS interface fi
file_dir = os.path.dirname(os.path.abspath(__file__))

class Floris:
    def __init__(self, file, n_turbines=3, wind_seed=0, width=None, height=None, n_samples=5, x_init:list=None,wd_no_bins:int=6,scipy_opt:dict={"maxiter": 400, "ftol": 1e-16, "eps": 0.005}):
        self.file = file #add the .yaml file. can be jensen or GCH model
        self.wind_seed = wind_seed
        self.n_turbines = n_turbines
        self. wd_no_bins = wd_no_bins # lf I will use 6 and HF I use 18
        self.scipy_opt = scipy_opt
        
        self.width = width if width is not None else 333.33 * n_turbines
        self.height = height if height is not None else 333.33 * n_turbines
        # Default polygon (covers entire area)
        self.boundaries = [[0.0, 0.0], [self.width, 0.0], [self.width, self.height], [0.0, self.height]]
        self.n_samples = n_samples
        self.wind_rng = np.random.RandomState(wind_seed)
        #self.wd, self.ws, self.freq = self._wind_random()
        self.wd, self.ws, self.freq = self._wind_weibull()
        self.fi = wfct.floris_interface.FlorisInterface(self.file)
        # Set number of turbines #TODO: use grid/Amalia layout as starting point
        if x_init is None:
            self.rand_layout_x = np.random.uniform(0.0, self.width, size=n_turbines)
            self.rand_layout_y = np.random.uniform(0.0, self.height, size=n_turbines)
        else:
            # note, these are the unnormalized values
            rand_layout_x = x_init[0:n_turbines]
            rand_layout_y = x_init[n_turbines:2*n_turbines]
            # above are the normalized values, need to unnormalize them. Note the below will
            # break when the boundaries are not 0,0 to width, height
            tmp_layout_x = np.array(rand_layout_x)*self.width
            tmp_layout_y = np.array(rand_layout_y)*self.height
            self.rand_layout_x = tmp_layout_x.tolist()
            self.rand_layout_y = tmp_layout_y.tolist()


        #self.fi.reinitialize_flow_field(layout_array=(rand_layout_x, rand_layout_y))
        self.fi.reinitialize(layout_x=self.rand_layout_x, layout_y=self.rand_layout_y, wind_directions=self.wd, wind_speeds=self.ws)

        
        # Scaling factor, set to 1 in order to avoid scaling. The initial Annual Energy
        # Production used for normalization in the optimization
        #self.aep_initial = 1
        #self.lo = wfct.optimization.scipy.layout.LayoutOptimization(self.fi, self.boundaries, self.wd, self.ws, self.freq, self.aep_initial)
        #self.lo = LayoutOptimizationScipy(self.fi, self.boundaries, self.wd, self.ws, self.freq, self.aep_initial)
        self.lo = LayoutOptimizationScipy(self.fi, self.boundaries, freq=self.freq, optOptions=self.scipy_opt) # TODO: need to pass freq, now considers equal dist.
        
        #self.lo = wfct.optimization.layout_optimization.layout_optimization_scipy.LayoutOptimizationScipy(self.fi, self.boundaries, self.wd, self.ws, self.freq, self.aep_initial)
        # Use the default minimum distance that floris themselves use.
        self.min_dist = self.lo.min_dist
        # logging.setLoggerClass(self.loggerclass)
        
    def _wind_random(self):
        # TODO: wind speed and direction dist can be taken from Padron et al. 2019
        # they have wind direction csv and use weibull for wind speed
        rng = self.wind_rng
        wd = np.arange(0.0, 360.0, 5.0)
        ws = 8.0 + rng.randn(1) * 0.5
        freq = (
            np.abs(
                np.sort(
                    np.random.randn(len(wd))
                )
            )
            .reshape( ( len(wd), len(ws) ) )
        )
        freq = freq / freq.sum()
        return wd, ws, freq #TODO: dont know what it is, need to check
    
    def _wind_weibull(self):
        """Generate wind speed and direction distribution using Weibull distribution."""
        # for HF
        # wd = np.linspace(0, 360, 18)
        # ws = np.linspace(0, 26, 14)

        # for LF
        wd = np.linspace(0, 360, self.wd_no_bins)
        #ws = np.linspace(0, 26, 5)
        rng = self.wind_rng
        #ws = 8.0 + rng.randn(1) * 0.5
        # TODO: taking constant wind speed now
        ws = np.array([8.0])

        wind_rose = wfct.wind_rose.WindRose()

        df = wind_rose.make_wind_rose_from_weibull(wd=wd, ws=ws)

        # if wind rose plot is needed
        wind_rose.plot_wind_rose()
        
        freq = df['freq_val'].to_numpy().reshape((len(wd), len(ws)))

        #return df['wd'].to_numpy(), df['ws'].to_numpy(), freq
        return df['wd'].to_numpy(), ws, freq

    # def evaluate_obj(self, x):
    #     if self.n_samples is None:
    #         obj = self.lo._AEP_layout_opt(x)
    #     else:
    #         obj = 0.0
    #         for _ in range(self.n_samples):
    #             # Resample wind speed
    #             self.ws = 8.0 + self.wind_rng.randn(len(self.wd)) * 0.5
    #             #self.lo = wfct.optimization.scipy.layout.LayoutOptimization(self.fi, self.boundaries, self.wd, self.ws, self.freq, self.aep_initial)
    #             self.lo = LayoutOptimizationScipy(self.fi, self.boundaries, self.wd, self.ws, self.freq, self.aep_initial)
    #             #obj += self.lo._AEP_layout_opt(x)
    #             obj += self.lo.obj_func(x)
    #         obj = obj / self.n_samples

    #     return obj
    
    def evaluate_obj(self, x):
            """Evaluate the objective function.

            This function takes normalized inputs between [0,1] and calculates the objective function value.
            The objective function value is the negative of the Annual Energy Production (AEP) and is normalized by self.aep_initial.
            The function internally performs unnormalization, change of coordinates, and reinitialization.

            Parameters
            ----------
            x : list
                inputs normalized between [0,1], [x_1 ... x_N, y_1 ... y_N]

            Returns
            -------
            obj : float
                The normalized AEP for the given layout.
            """
            # value in watt-hours
            
            # returns -1*AEP, also it is normalized by self.aep_initial (automatically adjusted, AEP with x_init). 
            # The fn does unnorm, change of corrdinates and reinit internally
            # TODO : integrate out the wind speed and direction with MC sampling
            obj = self.lo._obj_func(x) 

            # -- can add the below trick to make the constraint satisfied
            # taken from https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/windwake.py

            # c1 = self.lo._space_constraint(x) # returns -1*constraint
            # c2 = self.lo._distance_from_boundaries(x, self.boundaries)

            # # No power produced when constraints are violated.
            # if c1 < 0 or c2 < 0:
            #     return 0.0
        
            return obj
    
    def evaluate_space_constr(self, x:list)->float:
        """
        Evaluates the space constraint for a given input vector.

        Parameters:
            x (list): List of turbine locations in normalized coordinates.

        Returns:
            float: The value of the space constraint.
        """
        # # Constraint is satisfied when KS_constraint <= 0, if change is sign is needed, -1 can be multiplied
        c1 = self.lo._space_constraint(x) # returns -1*constraint
    

        #can add the boundary constraint too. See _distance_from_boundaries method in layout_optimization_scipy.py
        #return -c1, -c2 # added -1 to make constraint negative hwen its satisfied and positive when not satisfied
        return -c1
    
    def evaluate_distance_from_boundries_constr(self, x: list):
        """
        Evaluate the distance from boundaries constraint.

        Parameters:
        - x (list):List of turbine locations in normalized coordinates.

        Returns:
        - float: The mean of the normalized distance from boundaries constraint.
        """
        c = self.lo._distance_from_boundaries(x)
        # normalizing it with the width of the domain, also added negative sign constraint negative when it's satisfied and positive when not satisfied
        # TODO: think of a better way to normalize it
        c = -c / self.width
        c_mean = np.mean(c[c > 0])

        # return 0.0 if nan
        if np.isnan(c_mean):
            c_mean = 0.0

        return c_mean

    
    def get_AEP(self, x):
        """
        Calculates the Annual Energy Production (AEP) of the wind farm.

        Args:
            x (list): List of turbine locations in normalized coordinates.

        Returns:
            float: The AEP of the wind farm in MWh.
        """
        locs_unnorm = [
            self.lo._unnorm(valx, self.lo.xmin, self.lo.xmax)
            for valx in x[0 : self.lo.nturbs]
        ] + [
            self.lo._unnorm(valy, self.lo.ymin, self.lo.ymax)
            for valy in x[self.lo.nturbs : 2 * self.lo.nturbs]
        ]
        self.lo._change_coordinates(locs_unnorm)
        self.fi.reinitialize(
            layout_x=locs_unnorm[0 : self.lo.nturbs],
            layout_y=locs_unnorm[self.lo.nturbs : 2 * self.lo.nturbs],
        )
        # Compute turbine yaw angles using PJ's geometric code (if enabled)
        # todo : get_farm_AEP_wind_rose_class
        yaw_angles = self.lo._get_geoyaw_angles()
        return self.fi.get_farm_AEP(self.freq, yaw_angles=yaw_angles) / 1e6  # in MWh
    
    def scipy_optimize(self, normalize=True):
        """
        Performs optimization using scipy library.

        Parameters:
            normalize (bool): Flag indicating whether to normalize the optimization results.

        Returns:
            list: Normalized optimization results if normalize is True, otherwise unnormalized results.
        """
        

        # Run the optimization
        sol = self.lo.optimize()


        if normalize:
            sol_norm = [
            self.lo._norm(valx, self.lo.xmin, self.lo.xmax)
            for valx in sol[0]
        ] + [
            self.lo._norm(valy, self.lo.ymin, self.lo.ymax)
            for valy in sol[1]
        ]

        return sol_norm
    def get_unnorm_initial_final_values(self,x_init:list, x_opt:list):
        """Return list of unnormalized init values and a list of unnormalized final values"""
        # x is first half of the init_des vector
        x_initial = np.array(x_init[:len(x_init)//2])*self.width
        # y is second half of the init_des vector
        y_initial = np.array(x_init[len(x_init)//2:])*self.height

        x_init_unnorm = x_initial.tolist() + y_initial.tolist()

        x_final_ = np.array(x_opt[:len(x_opt)//2])*self.width
        y_final_ = np.array(x_opt[len(x_opt)//2:])*self.height
        x_opt = x_final_.tolist()
        y_opt = y_final_.tolist()

        # combine two lists to a single list named x_opt_unnorm
        x_opt_unnorm = x_opt + y_opt

        return x_init_unnorm, x_opt_unnorm


    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # returns a tuple of objective and constraint
        return self.evaluate_obj(*args, **kwds), self.evaluate_constr(*args, **kwds)
        


if __name__ == '__main__':
    # Initialize the FLORIS interface fi
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file = 'inputs/gch.yaml'
    floris = Floris(file, n_turbines=6)

    # testing optimization
    sol = floris.scipy_optimize()
    print(f'The optimized layout is {sol}')

    #x_test = [0, 0.2,0.4,0,0.2,0.4] # inputs normalized between [0,1], [x_1 ... x_N, y_1 ... y_N]
    #x_test = [0,0.2,0.4,0.6,0.8,0.95,0,0.2,0.4,0.6,0.8,0.95] # inputs normalized between [0,1], [x_1 ... x_N, y_1 ... y_N]
    x_test = [0,0.2,0.4,0.6,0.8,1.10,0,0.2,0.4,0.6,0.8,0.95]
    obj = floris.evaluate_obj(x_test)
    print(f'The normalized AEP for the given layout is {obj}')

    AEP = floris.get_AEP(x_test)
    print(f'The AEP for the given layout is {AEP} MWh')

    cons = floris.evaluate_constr(x_test)
    print(f'The constraint for the given layout is {cons}')