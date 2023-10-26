import json
import logging
import os
import shutil
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

import torch


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def compare_as_plot_vy(y_true, y_pred, lookUp):
    print("Vy MSE(model) : ", mean_absolute_error(y_pred, y_true))
    print("Vy MSE(LUT) : ", mean_absolute_error(lookUp, y_true))
    #print("Vy MSE(model) : ", mean_squared_error(y_pred, y_true))
    #print("Vy MSE(LUT) : ", mean_squared_error(lookUp, y_true))
    print("Vy Peak Error(model) : ", peak_error(y_pred, y_true))
    print("Vy Peak Error(LUT) : ", peak_error(lookUp, y_true))
    # Create a plot
    plt.figure(figsize=(20, 15))

    # Plot true data in blue
    plt.plot(range(len(y_true)), y_true, c='blue', label='True Data')

    # Plot predicted data in red
    plt.plot(range(len(y_pred)), y_pred, c='red', label='Predicted Data')

    plt.plot(range(len(lookUp)), lookUp, c='green', label='LookUpTable Data')

    # Add labels and legend
    plt.xlabel('time[0.01s]')
    plt.ylabel('Beta[rad]')
    plt.legend()

    # Show the plot
    plt.show()

    plt.figure(figsize=(20, 15))
    plt.plot(range(len(y_true)), np.abs(y_pred - y_true), c='red', label='error')
    plt.xlabel('time[0.01s]')
    plt.ylabel('error')
    plt.show()



def compare_as_plot_yr(y_true, y_pred, lookUp):
    print("Yr MSE(model) : ", mean_absolute_error(y_pred, y_true))
    print("Yr MSE(LUT) : ", mean_absolute_error(lookUp, y_true))
    #print("Yr MSE(model) : ", mean_squared_error(y_pred, y_true))
    #print("Yr MSE(LUT) : ", mean_squared_error(lookUp, y_true))

    print("Yr Peak Error(model) : ", peak_error(y_pred, y_true))
    print("Yr Peak Error(LUT) : ", peak_error(lookUp, y_true))
    # Create a plot
    plt.figure(figsize=(20, 15))

    # Plot true data in blue
    plt.plot(range(len(y_true)), y_true, c='blue', label='True Data')

    # Plot predicted data in red
    plt.plot(range(len(y_pred)), y_pred, c='red', label='Predicted Data')

    plt.plot(range(len(lookUp)), lookUp, c='green', label='LookUpTable Data')

    # Add labels and legend
    plt.xlabel('time[0.01s]')
    plt.ylabel('Yaw rate[rps]')
    plt.legend()

    # Show the plot
    plt.show()



def CfCr_as_plot(y_pred, Cf_look, Cr_look):
    Cf_nom = 253000 * 0.8
    Cr_nom = 295000

    # Create a plot
    plt.figure(figsize=(20, 15))
    #Cf_pred = 2 * Cf_nom * y_pred[:, 0] + 0.5 * Cf_nom
    #Cr_pred = 2 * Cr_nom * y_pred[:, 1] + 0.5 * Cr_nom

    Cf_pred = Cf_nom * y_pred[:, 0]
    Cr_pred = Cr_nom * y_pred[:, 1]

    Cf_look = Cf_look * Cf_nom
    Cr_look = Cr_look * Cr_nom

    # Plot true data in blue
    plt.plot(range(len(Cf_pred)), Cf_pred, c='blue', label='Cf')

    # Plot predicted data in red
    plt.plot(range(len(Cr_pred)), Cr_pred, c='red', label='Cr')

    plt.plot(range(len(Cf_look)), Cf_look, c='green', label='Cf_lookup')

    plt.plot(range(len(Cr_look)), Cr_look, c='brown', label='Cr_lookup')

    # Add labels and legend
    plt.xlabel('time[0.01s]')
    plt.ylabel('cornering stiffness')
    plt.legend()

    # Show the plot
    plt.show()


def bicycle_dVy1(Cf, Cr, Vx, Vy, Yaw, Sas):
    # Mf_nom = 904  # [Kg] 운전석에 1명 탔을때
    # Mr_nom = 619  # [Kg] 운전석에 1명 탔을때

    if Vx == 0:
        return 0;

    Mf_nom = 945  # [Kg] 운전석에 2명 탔을때
    Mr_nom = 728  # [Kg] 운전석에 2명탔을때

    M_nom = 2374
    #M_nom = Mf_nom + Mr_nom  # [kg]

    L = 3.010
    Lf_nom = (577+564)/(577+564+556+565)*L
    Lr_nom = L-Lf_nom

    #Lf_nom = 2.645 * Mr_nom / M_nom  # [m]
    #Lr_nom = 2.645 - Lf_nom  # [m]

    T = 0.01  # time step

    dVy = (-(Cf + Cr) / (M_nom * Vx) * Vy +
           ((Lr_nom * Cr - Lf_nom * Cf) / (M_nom * Vx) - Vx) * Yaw +
           Cf * Sas / M_nom) * T
    return dVy


def bicycle_Vy1(Cf, Cr, Vx, Vy, yr, Sas):
    # Mf_nom = 904  # [Kg] 운전석에 1명 탔을때
    # Mr_nom = 619  # [Kg] 운전석에 1명 탔을때

    Mf_nom = 945  # [Kg] 운전석에 2명 탔을때
    Mr_nom = 728  # [Kg] 운전석에 2명탔을때

    #if Vx == 0:
    #    return 0;

    #Cf_nom = 253000 * 0.8
    #Cr_nom = 295000

    Cf_nom = 168400
    Cr_nom = 276200

    M_nom = 2374
    # M_nom = Mf_nom + Mr_nom  # [kg]

    L = 3.010
    Lf_nom = (577+564)/(577+564+556+565)*L

    Lr_nom = L-Lf_nom

    # Lf_nom = 2.645 * Mr_nom / M_nom  # [m]
    # Lr_nom = 2.645 - Lf_nom  # [m]

    T = 0.01  # time step

    #Cf = 1.2 * Cf_nom * Cf + 0.05 * Cf_nom
    #Cr = 1.2 * Cr_nom * Cr + 0.05 * Cr_nom

    Cf = Cf_nom * Cf
    Cr = Cr_nom * Cr
    #if Vx.item() < 0.5:
    #    T = 0
    Vx = replace_under1_with_1(Vx)
    #Vx = replace_zeros_with_small_value(Vx)
    #Vy = replace_zeros_with_small_value(Vy)

    '''
    (1-(Cf(1,i-1) + Cr(1,i-1))*Ts / (M_nom * v0)) .* Vy_est_k_1 + ...
        ((Lr_nom * Cr(1,i-1) - Lf_nom * Cf(1,i-1)) / (M_nom * v0)- v0)*Ts .* yawrate_k_1 + ...
        Cf(1,i-1) .* sas_loss(1,i-1)*Ts / M_nom;
        '''

    Vy_next = (1-(Cf + Cr) * T / (M_nom * Vx)) * Vy + \
              ((Lr_nom * Cr - Lf_nom * Cf) / (M_nom * Vx) - Vx) * T * yr + \
              Cf * Sas * T / M_nom

    dVy_dCf = (Vy - Lf_nom * yr) / (M_nom * Vx) * T + Sas * T / M_nom
    dVy_dCr = (Vy + Lr_nom * yr) / (M_nom * Vx) * T

    #print("1", (1-(Cf + Cr) * T / (M_nom * Vx)) * Vy)
    #print("2", ((Lr_nom * Cr - Lf_nom * Cf) / (M_nom * Vx) - Vx) * T * yr)
    #print("3", Cf * Sas * T / M_nom)

    return Vy_next, dVy_dCf, dVy_dCr


def bicycle_Vy2(Cf, Cr, Vx, Vy, yr, Sas):
    # Mf_nom = 904  # [Kg] 운전석에 1명 탔을때
    # Mr_nom = 619  # [Kg] 운전석에 1명 탔을때

    Mf_nom = 945  # [Kg] 운전석에 2명 탔을때
    Mr_nom = 728  # [Kg] 운전석에 2명탔을때

    # if Vx == 0:
    #    return 0;

    #Cf_nom = 253000 * 0.8
    #Cr_nom = 295000

    Cf_nom = 168400
    Cr_nom = 276200

    M_nom = 2374
    # M_nom = Mf_nom + Mr_nom  # [kg]

    L = 3.010
    Lf_nom = (577 + 564) / (577 + 564 + 556 + 565) * L

    Lr_nom = L - Lf_nom

    # Lf_nom = 2.645 * Mr_nom / M_nom  # [m]
    # Lr_nom = 2.645 - Lf_nom  # [m]

    T = 0.01  # time step

    Cf = Cf_nom * Cf
    Cr = Cr_nom * Cr
    # if Vx.item() < 0.5:
    #    T = 0
    Vx = replace_under1_with_1(Vx)
    # Vx = replace_zeros_with_small_value(Vx)
    # Vy = replace_zeros_with_small_value(Vy)

    '''
    (1-(Cf(1,i-1) + Cr(1,i-1))*Ts / (M_nom * v0)) .* Vy_est_k_1 + ...
        ((Lr_nom * Cr(1,i-1) - Lf_nom * Cf(1,i-1)) / (M_nom * v0)- v0)*Ts .* yawrate_k_1 + ...
        Cf(1,i-1) .* sas_loss(1,i-1)*Ts / M_nom;
        '''

    Vy_next = (1 - (Cf + Cr) * T / (M_nom * Vx)) * Vy + \
              ((Lr_nom * Cr - Lf_nom * Cf) / (M_nom * Vx) - Vx) * T * yr + \
              Cf * Sas * T / M_nom

    #dVy_dCf = (Vy - Lf_nom * yr) / (M_nom * Vx) * T + Sas * T / M_nom
    #dVy_dCr = (Vy + Lr_nom * yr) / (M_nom * Vx) * T

    # print("1", (1-(Cf + Cr) * T / (M_nom * Vx)) * Vy)
    # print("2", ((Lr_nom * Cr - Lf_nom * Cf) / (M_nom * Vx) - Vx) * T * yr)
    # print("3", Cf * Sas * T / M_nom)

    return Vy_next


def bicycle_beta(Cf, Cr, Vx, Beta, yr, Sas):
    Cf_nom = 168400
    Cr_nom = 276200

    M_nom = 2374
    # M_nom = Mf_nom + Mr_nom  # [kg]

    L = 3.010
    Lf_nom = (577 + 564) / (577 + 564 + 556 + 565) * L

    Lr_nom = L - Lf_nom

    # Lf_nom = 2.645 * Mr_nom / M_nom  # [m]
    # Lr_nom = 2.645 - Lf_nom  # [m]

    T = 0.01  # time step

    Cf = Cf_nom * Cf
    Cr = Cr_nom * Cr
    # if Vx.item() < 0.5:
    #    T = 0
   # Vx = replace_under1_with_1(Vx)

    beta_next = - yr * T + (1 -(Cf + Cr) * T / (M_nom * Vx)) * Beta + \
              (- Lf_nom * Cf + Lr_nom * Cr) / (M_nom * Vx * Vx) * T * yr + \
              Cf * Sas * T / (M_nom * Vx)

    return beta_next

def bicycle_yr1(Cf, Cr, Vx, Vy, yr, Sas):
    # Mf_nom = 904  # [Kg] 운전석에 1명 탔을때
    # Mr_nom = 619  # [Kg] 운전석에 1명 탔을때

    #if Vx == 0:
    #    return 0;

    Mf_nom = 945  # [Kg] 운전석에 2명 탔을때
    Mr_nom = 728  # [Kg] 운전석에 2명탔을때

    M_nom = 2374
    # M_nom = Mf_nom + Mr_nom  # [kg]

    L = 3.010
    Lf_nom = (577+564)/(577+564+556+565)*L
    Lr_nom = L - Lf_nom

    # Lf_nom = 2.645 * Mr_nom / M_nom  # [m]
    # Lr_nom = 2.645 - Lf_nom  # [m]

    #Cf_nom = 253000 * 0.8
    #Cr_nom = 295000

    Cf_nom = 168400
    Cr_nom = 276200

    J_nom = 5263
    #J_nom = 1 * 3122 + 102 * (Lf_nom ** 2 + Lr_nom ** 2) # 3122kgm^2 from LM Carsim par file (sprung mass inertia)

    T = 0.01  # time step

    # => remove minimum (2, 0.5) => (1.2)
    #Cf = 1.2 * Cf_nom * Cf + 0.05 * Cf_nom
    #Cr = 1.2 * Cr_nom * Cr + 0.05 * Cr_nom
    Cf = Cf_nom * Cf
    Cr = Cr_nom * Cr

    #if Vx.item() < 1:
    #    Vx = replace_zeros_with_small_value(Vx, 1)

    Vx = replace_under1_with_1(Vx)
    #Vx = replace_zeros_with_small_value(Vx)
    #Vy = replace_zeros_with_small_value(Vy)

    #Cf = 2 * 1.2148 * 10 ** 5 * Cf + 0.5 * 1.2715 * 10 ** 5
    #Cr = 2 * 1.2148 * 10 ** 5 * Cr + 0.5 * 1.2715 * 10 ** 5

    '''
    (-(Lf_nom*Cf(1,i-1) - Lr_nom*Cr(1,i-1))*Ts / (J_nom * v0)) .* Vy_est_k_1 + ...
        (1-((Lf_nom*Lf_nom*Cf(1,i-1) + Lr_nom*Lr_nom*Cr(1,i-1)) / (J_nom * v0))*Ts) .* yawrate_k_1 + ...
        Lf_nom*Cf(1,i-1) .* sas_loss(1,i-1)*Ts / J_nom;
        '''

    yr_next = (-(Lf_nom * Cf - Lr_nom * Cr) * T / (J_nom * Vx)) * Vy + \
              (1 - ((Lf_nom * Lf_nom * Cf + Lr_nom * Lr_nom * Cr) / (J_nom * Vx)) * T) * yr + \
              Lf_nom * Cf * Sas * T / J_nom

    dYr_dCf = -(Lf_nom * Vy + (Lf_nom ** 2) * yr) / (J_nom * Vx) * T + Lf_nom * Sas * T / J_nom
    dYr_dCr = (Lr_nom * Vy - (Lr_nom ** 2) * yr) / (J_nom * Vx) * T

    return yr_next, dYr_dCf, dYr_dCr


def bicycle_yr2(Cf, Cr, Vx, Vy, yr, Sas):
    # Mf_nom = 904  # [Kg] 운전석에 1명 탔을때
    # Mr_nom = 619  # [Kg] 운전석에 1명 탔을때

    # if Vx == 0:
    #    return 0;

    Mf_nom = 945  # [Kg] 운전석에 2명 탔을때
    Mr_nom = 728  # [Kg] 운전석에 2명탔을때

    M_nom = 2374
    # M_nom = Mf_nom + Mr_nom  # [kg]

    L = 3.010
    Lf_nom = (577 + 564) / (577 + 564 + 556 + 565) * L
    Lr_nom = L - Lf_nom

    # Lf_nom = 2.645 * Mr_nom / M_nom  # [m]
    # Lr_nom = 2.645 - Lf_nom  # [m]

    #Cf_nom = 253000 * 0.8
    #Cr_nom = 295000
    Cf_nom = 168400
    Cr_nom = 276200
    J_nom = 5263
    # J_nom = 1 * 3122 + 102 * (Lf_nom ** 2 + Lr_nom ** 2) # 3122kgm^2 from LM Carsim par file (sprung mass inertia)

    T = 0.01  # time step

    Cf = Cf_nom * Cf
    Cr = Cr_nom * Cr

    # if Vx.item() < 1:
    #    Vx = replace_zeros_with_small_value(Vx, 1)

    Vx = replace_under1_with_1(Vx)
    # Vx = replace_zeros_with_small_value(Vx)
    # Vy = replace_zeros_with_small_value(Vy)

    # Cf = 2 * 1.2148 * 10 ** 5 * Cf + 0.5 * 1.2715 * 10 ** 5
    # Cr = 2 * 1.2148 * 10 ** 5 * Cr + 0.5 * 1.2715 * 10 ** 5

    '''
    (-(Lf_nom*Cf(1,i-1) - Lr_nom*Cr(1,i-1))*Ts / (J_nom * v0)) .* Vy_est_k_1 + ...
        (1-((Lf_nom*Lf_nom*Cf(1,i-1) + Lr_nom*Lr_nom*Cr(1,i-1)) / (J_nom * v0))*Ts) .* yawrate_k_1 + ...
        Lf_nom*Cf(1,i-1) .* sas_loss(1,i-1)*Ts / J_nom;
        '''

    yr_next = (-(Lf_nom * Cf - Lr_nom * Cr) * T / (J_nom * Vx)) * Vy + \
              (1 - ((Lf_nom * Lf_nom * Cf + Lr_nom * Lr_nom * Cr) / (J_nom * Vx)) * T) * yr + \
              Lf_nom * Cf * Sas * T / J_nom

    #dYr_dCf = -(Lf_nom * Vy + (Lf_nom ** 2) * yr) / (J_nom * Vx) * T + Lf_nom * Sas * T / J_nom
    #dYr_dCr = (Lr_nom * Vy - (Lr_nom ** 2) * yr) / (J_nom * Vx) * T

    return yr_next



def lookup_table_cf(ay):
    #C_look = [0.000, 1.245, 1.79, 2.065, 3.014, 4.198, 5.414, 5.8, 6.200, 6.4, 7.056, 7.607, 7.954, 8.156, 8.340, 8.7, 9.010, 9.300, 9.500, 10.00, 11.76]

    C_look = [0.00, 0.63157895, 1.26315789, 1.89473684, 2.52631579, 3.15789474, 3.78947368, 4.42105263, 5.05263158, 5.68421053, 6.31578947, 6.94736842, 7.57894737, 8.21052632, 8.84210526, 9.47368421, 10.10526316, 10.73684211, 11.36842105, 12.00]

    #Cf_look = [1.00, 1.00, 0.95, 0.93, 0.9 * 0.96, 0.88 * 0.96, 0.86 * 0.96, 0.84 * 0.95, 0.80 * 0.95, 0.74, 0.715, 0.710, 0.70, 0.680, 0.65, 0.64, 0.6, 0.55, 0.48, 0.4, 0.4]

    Cf_look = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.999974, 0.938764, 0.934219, 0.885539, 0.885530, 0.847389, 0.795243, 0.736241, 0.553619, 0.458934, 0.391231, 0.070288, 0.070288, 0.070288]

    # Create an interpolation function with linear interpolation
    interpolation_function = interp1d(C_look, Cf_look, kind='linear', fill_value='extrapolate')

    # Perform interpolation/extrapolation at the new data points
    interpolated_values = interpolation_function(ay)

    return interpolated_values


def lookup_table_cr(ay):
    #C_look = [0.000, 1.245, 1.79, 2.065, 3.014, 4.198, 5.414, 5.8, 6.200, 6.4, 7.056, 7.607, 7.954, 8.156, 8.340, 8.7, 9.010, 9.300, 9.500, 10.00, 11.76]

    C_look = [0.00, 0.63157895, 1.26315789, 1.89473684, 2.52631579, 3.15789474, 3.78947368, 4.42105263, 5.05263158, 5.68421053, 6.31578947, 6.94736842, 7.57894737, 8.21052632, 8.84210526, 9.47368421, 10.10526316, 10.73684211, 11.36842105, 12.00]

    #Cr_look = [1.00, 1.00, 1.00, 1.00, 1, 0.98, 0.95, 0.93, 0.93, 0.93, 0.900, 0.870, 0.85, 0.8, 0.75, 0.70, 0.62, 0.59, 0.58, 0.56, 0.54]

    Cr_look = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.928755, 0.928755, 0.882532, 0.817083, 0.75571, 0.622662, 0.404081, 0.404081, 0.155547, 0.155547, 0.155547]
    # Create an interpolation function with linear interpolation
    interpolation_function = interp1d(C_look, Cr_look, kind='linear', fill_value='extrapolate')

    # Perform interpolation/extrapolation at the new data points
    interpolated_values = interpolation_function(ay)

    return interpolated_values


def replace_zeros_with_small_value(tensor, small_value=1):
    # Create a condition tensor where the elements are True where the input tensor is 0, and False otherwise
    condition = tensor == 0

    # Use torch.where to replace elements where the condition is True (i.e., where the tensor is 0)
    # with the small_value, and leave the other elements unchanged
    result = torch.where(condition, torch.tensor(small_value), tensor)

    return result


def replace_under1_with_1(tensor, small_value=2):
    # Create a condition tensor where the elements are True where the input tensor is 0, and False otherwise
    condition = tensor < small_value

    # Use torch.where to replace elements where the condition is True (i.e., where the tensor is 0)
    # with the small_value, and leave the other elements unchanged
    result = torch.where(condition, torch.tensor(small_value), tensor)

    return result


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            #ave_grads.append(p.grad.abs().mean())
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    #plt.show()


def plot_grad_flow_now():
    plt.show()


def peak_error(g1, g2):
    peaks_graph, _ = find_peaks(g2)
    peak_values_graph1 = g1[peaks_graph]
    peak_values_graph2 = g2[peaks_graph]

    error = np.mean(np.abs(peak_values_graph1 - peak_values_graph2))


    return error