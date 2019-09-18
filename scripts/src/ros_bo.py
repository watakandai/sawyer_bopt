import copy
import time
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import GPyOpt
import collections
import numpy as np
import time
import csv
import rospy
from GPyOpt.util.general import best_value, normalize
from GPyOpt.util.duplicate_manager import DuplicateManager
from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.core.task.cost import CostModel
from GPyOpt.optimization.acquisition_optimizer import ContextManager
from GPyOpt.experiment_design import initial_design

try:
    from GPyOpt.plotting.plots_bo import plot_acquisition, plot_convergence
except:
    pass


class RosBO(object):
    """
    Runner of Bayesian optimization loop. This class wraps the optimization loop around the different handlers.
    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param de_duplication: GPyOpt DuplicateManager class. Avoids re-evaluating the objective at previous, pending or infeasible locations (default, False).
    """


    def __init__(self, model, space, acquisition, evaluator, X_init, Y_init=None, cost = None, normalize_Y = True, model_update_interval = 1, de_duplication = False, initial_design_numdata = 2):
        self.model = model
        self.space = space
        self.acquisition = acquisition
        self.evaluator = evaluator
        self.normalize_Y = normalize_Y
        self.model_update_interval = model_update_interval
        self.X = X_init
        self.Y = Y_init
        self.cost = CostModel(cost)
        self.normalization_type = 'stats' ## not added in the API
        self.de_duplication = de_duplication
        self.initial_design_numdata = initial_design_numdata
        self.context = None
        self.num_acquisitions = 0
        self.sample_for_pred = 50
        self.prev_m = self.prev_v = np.inf*np.ones((self.sample_for_pred, self.sample_for_pred, self.sample_for_pred))
        self.X_for_pred = self.X_for_prediction()

    def X_for_prediction(self):
        bounds = self.acquisition.space.get_bounds()
        # Sample data
        X1 = np.linspace(bounds[0][0], bounds[0][1], self.sample_for_pred)
        X2 = np.linspace(bounds[1][0], bounds[1][1], self.sample_for_pred)
        X3 = np.linspace(bounds[2][0], bounds[2][1], self.sample_for_pred)
        x1, x2, x3 = np.meshgrid(X1, X2, X3)
        return np.hstack((
            x1.reshape(self.sample_for_pred**3,1), 
            x2.reshape(self.sample_for_pred**3,1), 
            x3.reshape(self.sample_for_pred**3,1)
        ))

    def init(self, max_iter=0, context=None, verbosity=False):
        """
        Runs Bayesian Optimization 
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        """
        # --- Save the options to print and save the results
        self.max_iter = max_iter
        self.verbosity = verbosity
        self.context = context

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = initial_design('random', self.space, 1)
        self.X = self.suggested_sample
        self.Y_new = self.Y
        return np.array(self.suggested_sample[0,:])

    def update(self, Y_new):
        """
        Update Gaussian Models
       :param Y_new: new data
        """
        if self.Y is None:
            self.Y = np.array([-Y_new])
        else:
            self.Y = np.vstack((self.Y, -Y_new))

        self._compute_results()
        
        # --- Update model
        if self.Y.shape[0]>=2:
            try:
                self._update_model(self.normalization_type)
            except np.linalg.linalg.LinAlgError:
                return True

        if self.num_acquisitions >= self.max_iter:
            return True


    def step(self):
        """
        Runs Bayesian Optimization every loop
        """
        if self.Y.shape[0]<self.initial_design_numdata:
            self.suggested_sample = initial_design('random', self.space, 1)
        else:
            self.suggested_sample = self._compute_next_evaluations()

        self.X = np.vstack((self.X,self.suggested_sample))

        # --- Update current evaluation time and function evaluations
        self.num_acquisitions += 1

        if self.verbosity:
            print("num acquisition: {}".format(self.num_acquisitions))

        return np.array(self.suggested_sample[0,:])


    def _compute_results(self):
        """
        Computes the optimum and its value.
        """
        self.Y_best = best_value(self.Y)
        self.x_opt = self.X[np.argmin(self.Y),:]
        self.fx_opt = np.min(self.Y)
        self.distance = self._compute_distance_betw_consecutive_x()

    def _compute_distance_betw_consecutive_x(self):
        n = self.X.shape[0]
        aux = (self.X[1:n,:]-self.X[0:n-1,:])**2
        return np.sqrt(aux.sum(axis=1))

    def _distance_last_evaluations(self):
        """
        Computes the distance between the last two evaluations.
        """
        if self.X.shape[0] < 2:
            # less than 2 evaluations
            return np.inf
        return np.sqrt(np.sum((self.X[-1, :] - self.X[-2, :]) ** 2))

    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None):
        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """

        ## --- Update the context if any
        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)

        ### --- Activate de_duplication
        if self.de_duplication:
            duplicate_manager = DuplicateManager(space=self.space, zipped_X=self.X, pending_zipped_X=pending_zipped_X, ignored_zipped_X=ignored_zipped_X)
        else:
            duplicate_manager = None

        ### We zip the value in case there are categorical variables
        return self.space.zip_inputs(self.evaluator.compute_batch(duplicate_manager=duplicate_manager, context_manager= self.acquisition.optimizer.context_manager))

    def _update_model(self, normalization_type='stats'):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """
        if self.num_acquisitions % self.model_update_interval == 0:

            # input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.space.unzip_inputs(self.X)

            # Y_inmodel is the output that goes into the model
            if self.normalize_Y:
                Y_inmodel = normalize(self.Y, normalization_type)
            else:
                Y_inmodel = self.Y

            self.model.updateModel(X_inmodel, Y_inmodel, None, None)

    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()

    def compute_mean_var(self):
        # Data: mean, var
        m, v = self.model.model.predict(self.X_for_pred)
        m = m.reshape(self.sample_for_pred,self.sample_for_pred,self.sample_for_pred)
        v = v.reshape(self.sample_for_pred,self.sample_for_pred,self.sample_for_pred)
        return m, v

    def mean_var_difference(self):
        if self.Y.shape[0] >= 3:
            curr_m, curr_v = self.compute_mean_var()
            diff_m = curr_m - self.prev_m
            diff_v = curr_v - self.prev_v
            self.prev_m = curr_m
            self.prev_v = curr_v
            return diff_m, diff_v
        else:
            return self.prev_m, self.prev_v

    def plot_acquisition_frame(self, ax1, ax2, ax3):
        """
        Plots the model and the acquisition function.
            if self.input_dim = 1: Plots data, mean and variance in one plot and the acquisition function in another plot
            if self.input_dim = 2: as before but it separates the mean and variance of the model in two different plots
        :param filename: name of the file where the plot is saved
        """
        model_to_plot = self.model

        return self.plot2frames_acquisition(ax1, ax2, ax3,
                                            self.acquisition.space.get_bounds(),
                                            model_to_plot.model.X.shape[1],
                                            model_to_plot.model,
                                            model_to_plot.model.X,
                                            model_to_plot.model.Y,
                                            self.acquisition.acquisition_function, 
                                            self._compute_next_evaluations())

    def plot2frames_acquisition(self, ax1, ax2, ax3, bounds, input_dim, model, Xdata, Ydata, acquisition_function, suggested_sample):
        '''
        Plot Average, STD, Acquisiton Function
        If colormap is needed, add these before animation loop
            ax = fig.add_subplot()
            contourplot = ax.contourf(X1, X2, Z, leves, vmin, vmax)
            plt.colorbar(contourplot, ax=ax)
        '''
        X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
        X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))
        acqu = acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        acqu_normalized = acqu_normalized.reshape((200,200))
        m, v = model.predict(X)

        ## 
        p1 = ax1.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
        im = ax1.contourf(X1, X2, m.reshape(200,200),100, vmin=0.0, vmax=1.0) # returns QuadContourSet
        cb1 = im.collections # returns collections of QuadContourSet
        #ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_title('Posterior mean')
        ax1.set_xlim(bounds[0][0],bounds[0][1]) 
        ax1.set_ylim(bounds[1][0],bounds[1][1])

        ##
        x = np.sqrt(v.reshape(200,200))
        p2 = ax2.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
        im = ax2.contourf(X1, X2, np.sqrt(v.reshape(200,200)),100, vmin=0.0, vmax=1.0)
        cb2 = im.collections # returns collections of QuadContourSet
        #ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_title('Posterior Std.')
        ax2.set_xlim(bounds[0][0],bounds[0][1]) 
        ax2.set_ylim(bounds[1][0],bounds[1][1])

        p3 = ax3.plot(suggested_sample[:,0],suggested_sample[:,1],'k.', markersize=10)
        im = ax3.contourf(X1, X2, acqu_normalized, 100, vmin=0.0, vmax=1.0)
        cb3 = im.collections # returns collections of QuadContourSet
        ax3.set_xlabel('X1')
        ax3.set_ylabel('X2')
        ax3.set_title('Acquisition function')
        return p1+cb1 + p2+cb2 + p3+cb3, np.average(acqu_normalized.reshape((200*200,1)))

    def calculate_for_3d_plot(self, num):
        bounds = self.acquisition.space.get_bounds()
        # Sample data
        X1 = np.linspace(bounds[0][0], bounds[0][1], num)
        X2 = np.linspace(bounds[1][0], bounds[1][1], num)
        X3 = np.linspace(bounds[2][0], bounds[2][1], num)
        x1, x2, x3 = np.meshgrid(X1, X2, X3)
        X = np.hstack((x1.reshape(num**3,1),x2.reshape(num**3,1), x3.reshape(num**3,1)))
        # Data: mean, var
        m, v = self.model.model.predict(X)
        m = m.reshape(num,num,num)
        v = v.reshape(num,num,num)
        acqu = self.acquisition.acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        acqu_normalized = acqu_normalized.reshape((num,num,num))
        next_sample = self._compute_next_evaluations()

        return X1,X2,X3, m,v,acqu_normalized, next_sample 

    def plot_3d_frame(self, ax, i_trial, i_z, evaluate_type, X1,X2,X3, m,v,acqu, next_sample):
        # plot
        Xdata = self.model.model.X
        zlen = X3[-1]-X3[0]
        if evaluate_type is 1:
            tx = [ax.text(X1[0], X2[0], X3[-1]+0.1*zlen, 'Trial %i'%(i_trial), transform=ax.transAxes)]
            pl = ax.plot(Xdata[:,0], Xdata[:,1], Xdata[:,2], 'r.', markersize=10, label=u'Observations')
            im = ax.contourf(X1, X2, 1-m[:,:,i_z], vmin=0.0, vmax=1.0, zdir='z', offset=X3[i_z], 
                            levels=np.linspace(0.0,1.0,100, endpoint=True), alpha=0.05) # returns QuadContourSet
            cb = im.collections # returns collections of QuadContourSet
            if i_trial is 1 and i_z is 0:
                plt.colorbar(im, ax=ax, shrink=0.6, alpha=0.5)

        if evaluate_type is 2:
            tx = [ax.text(X1[0], X2[0], X3[-1]+0.1*zlen, 'Trial %i'%(i_trial), transform=ax.transAxes)]
            pl = ax.plot(Xdata[:,0], Xdata[:,1], Xdata[:,2], 'r.', markersize=10, label=u'Observations')
            im = ax.contourf(X1, X2, v[:,:,i_z], vmin=0.0, vmax=1.0, zdir='z', offset=X3[i_z], 
                            levels=np.linspace(0.0,1.0,100, endpoint=True), alpha=0.05) # returns QuadContourSet
            cb = im.collections # returns collections of QuadContourSet
            if i_trial is 1 and i_z is 0:
                plt.colorbar(im, ax=ax, shrink=0.6, alpha=0.5)
        
        if evaluate_type is 3:
            tx = [ax.text(X1[0], X2[0], X3[-1]+0.1*zlen, 'Trial %i'%(i_trial), transform=ax.transAxes)]
            pl = ax.plot(next_sample[:,0], next_sample[:,1], next_sample[:,2], 'k.', markersize=10)
            im = ax.contourf(X1, X2, acqu[:,:,i_z], vmin=0.0, vmax=1.0, zdir='z', offset=X3[i_z], 
                            levels=np.linspace(0.0,1.0,100, endpoint=True), alpha=0.05) # returns QuadContourSet
            cb = im.collections # returns collections of QuadContourSet
            if i_trial is 1 and i_z is 0:
                plt.colorbar(im, ax=ax, shrink=0.6, alpha=0.5)
            
        return tx + pl + cb

