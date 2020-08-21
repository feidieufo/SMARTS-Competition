import numpy as np
from user_config import DEFAULT_DATA_DIR, FORCE_DATESTAMP, \
                               DEFAULT_SHORTHAND, WAIT_BEFORE_LAUNCH
import time
import os.path as osp

# Can be used to convert rewards into discounted returns:
# ret[i] = sum of t = i to T of gamma^(t-i) * rew[t]
def discount_path(path, gamma):
    '''
    Given a "path" of items x_1, x_2, ... x_n, return the discounted
    path, i.e.
    X_1 = x_1 + h*x_2 + h^2 x_3 + h^3 x_4
    X_2 = x_2 + h*x_3 + h^2 x_4 + h^3 x_5
    etc.
    Can do (more efficiently?) w SciPy. Python here for readability
    Inputs:
    - path, list/tensor of floats
    - h, discount rate
    Outputs:
    - Discounted path, as above
    '''
    curr = 0
    rets = []
    for i in range(len(path)):
        curr = curr*gamma + path[-1-i]
        rets.append(curr)
    rets =  np.stack(list(reversed(rets)), 0)
    return rets


def get_path_indices(not_dones):
    """
    Returns list of tuples of the form:
        (agent index, time index start, time index end + 1)
    For each path seen in the not_dones array of shape (# agents, # time steps)
    E.g. if we have an not_dones of composition:
    tensor([[1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]], dtype=torch.uint8)
    Then we would return:
    [(0, 0, 3), (0, 3, 10), (1, 0, 3), (1, 3, 5), (1, 5, 9), (1, 9, 10)]
    """
    indices = []
    num_timesteps = not_dones.shape[0]
    last_index = 0
    for i in range(num_timesteps):
        if not_dones[i] == 0.:
            indices.append((last_index, i + 1))
            last_index = i + 1
    if last_index != num_timesteps:
        indices.append((last_index, num_timesteps))
    return indices

def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.

    If no seed is given and datestamp is false, 

    ::

        output_dir = data_dir/exp_name

    If a seed is given and datestamp is false,

    ::

        output_dir = data_dir/exp_name/exp_name_s[seed]

    If datestamp is true, amend to

    ::

        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]

    You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in 
    ``spinup/user_config.py``. 

    Args:

        exp_name (string): Name for experiment.

        seed (int): Seed for random number generators used by experiment.

        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``.

        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.

    Returns:

        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])
    
    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), 
                         exp_name=exp_name)
    return logger_kwargs