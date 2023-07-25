import json
import logging
import os
import shutil
import matplotlib.pyplot as plt

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


def compare_as_plot(y_true, y_pred):
    print(y_true.shape, y_pred.shape)
    # Create a plot
    plt.figure(figsize=(20, 15))

    # Plot true data in blue
    plt.plot(range(len(y_true)), y_true, c='blue', label='True Data')

    # Plot predicted data in red
    plt.plot(range(len(y_pred)), y_pred, c='red', label='Predicted Data')

    # Add labels and legend
    plt.xlabel('time[0.01s]')
    plt.ylabel('dVy')
    plt.legend()

    # Show the plot
    plt.show()


def CfCr_as_plot(y_pred):
    print(y_pred.shape)
    # Create a plot
    plt.figure(figsize=(20, 15))

    # Plot true data in blue
    plt.plot(range(len(y_pred[:, 0])), y_pred[:, 0], c='blue', label='Cf')

    # Plot predicted data in red
    plt.plot(range(len(y_pred[:, 1])), y_pred[:, 1], c='red', label='Cr')

    # Add labels and legend
    plt.xlabel('time[0.01s]')
    plt.ylabel('cornering stiffness')
    plt.legend()

    # Show the plot
    plt.show()


def bicycle_dVy1(Cf, Cr, Vx, Vy, Yaw, Sas):
    # Mf_nom = 904  # [Kg] 운전석에 1명 탔을때
    # Mr_nom = 619  # [Kg] 운전석에 1명 탔을때

    Mf_nom = 945  # [Kg] 운전석에 2명 탔을때
    Mr_nom = 728  # [Kg] 운전석에 2명탔을때

    M_nom = Mf_nom + Mr_nom  # [kg]

    Lf_nom = 2.645 * Mr_nom / M_nom  # [m]
    Lr_nom = 2.645 - Lf_nom  # [m]

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

    M_nom = Mf_nom + Mr_nom  # [kg]

    Lf_nom = 2.645 * Mr_nom / M_nom  # [m]
    Lr_nom = 2.645 - Lf_nom  # [m]

    T = 0.01  # time step

    #Cf = 2 * 1.2148 * 10 ** 5 * Cf + 0.5 * 1.2715 * 10 ** 5
    #Cr = 2 * 1.2148 * 10 ** 5 * Cr + 0.5 * 1.2715 * 10 ** 5

    '''
    (1-(Cf(1,i-1) + Cr(1,i-1))*Ts / (M_nom * v0)) .* Vy_est_k_1 + ...
        ((Lr_nom * Cr(1,i-1) - Lf_nom * Cf(1,i-1)) / (M_nom * v0)- v0)*Ts .* yawrate_k_1 + ...
        Cf(1,i-1) .* sas_loss(1,i-1)*Ts / M_nom;
        '''

    Vy_next = (1-(Cf + Cr) * T / (M_nom * Vx)) * Vy + \
              ((Lr_nom * Cr - Lf_nom * Cf) / (M_nom * Vx) - Vx) * T * yr + \
              Cf * Sas * T / M_nom

    return Vy_next


def bicycle_yr1(Cf, Cr, Vx, Vy, yr, Sas):
    # Mf_nom = 904  # [Kg] 운전석에 1명 탔을때
    # Mr_nom = 619  # [Kg] 운전석에 1명 탔을때

    Mf_nom = 945  # [Kg] 운전석에 2명 탔을때
    Mr_nom = 728  # [Kg] 운전석에 2명탔을때

    M_nom = Mf_nom + Mr_nom  # [kg]

    Lf_nom = 2.645 * Mr_nom / M_nom  # [m]
    Lr_nom = 2.645 - Lf_nom  # [m]

    J_nom = 1 * 3122 + 102 * (Lf_nom ** 2 + Lr_nom ** 2) # 3122kgm^2 from LM Carsim par file (sprung mass inertia)

    T = 0.01  # time step

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

    return yr_next