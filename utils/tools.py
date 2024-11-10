import sys, os, time, uuid
import numpy as np
import gurobipy as grb

# unique stamp for file name
def unique_stamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '_' + str(uuid.uuid4()).replace('-', '')

# progress bar
def update_progress(progress, start_time, method='', curr_exp_no=1, total_exps=1):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    
    curr_time = time.time()
    duration = curr_time - start_time
    text = "\r{}({:d}/{:d}) Progress: [{}] {:02.2f}% Duration: {:02.0f}h-{:02.0f}min-{:02.0f}s ".format(method, curr_exp_no, total_exps, "#"*block + "-"*(barLength-block), progress*100, duration//3600, (duration%3600)//60, (duration%3600)%60)
    sys.stdout.write(text)
    sys.stdout.flush()

# file operations
def read_from_file(file_addr):
    f = open(file_addr, 'r', encoding='utf-8')
    text = f.read()
    f.close()
    return text

def write_to_file(file_addr, text):
    f = open(file_addr, 'a', encoding='utf-8')
    f.write(text)
    f.close()

def save_ExpSettings(log_dir, data_addr, data_model_info, problem_dict, method, param_dict):
    write_to_file(log_dir, '----\n# DATA')
    write_to_file(log_dir, '\nData dir: '+data_addr)
    write_to_file(log_dir, '\n'+ data_model_info)

    write_to_file(log_dir, '\n----')
    write_to_file(log_dir, '\n# PROBLEM')
    for key, value in problem_dict.items():
        write_to_file(log_dir, f'\n{key} = {value}')

    write_to_file(log_dir, '\n----')
    write_to_file(log_dir, '\n# PARAMETERS')
    write_to_file(log_dir, f'\nmethod = {method}')
    for key, value in param_dict.items():
        write_to_file(log_dir, f'\n{key} = {value}')
    write_to_file(log_dir, '\n----\n')

# ops checking folder names
def check_folder_syntax(folder_dir):
    return folder_dir if folder_dir[-1] == '/' else folder_dir + '/'

def check_folder_syntaxandexistence(target_path):
    if os.path.exists(target_path):
        target_path = check_folder_syntax(target_path)[:-1]
        print("The path [{}] exists.".format(target_path))
        target_path += f'-{unique_stamp()}'
        print("Replaced by [{}].".format(target_path))
    target_path = check_folder_syntax(target_path)
    return target_path

def check_file_existence(target_file_name, form='pdf'):
    """
    target_file_name: the name in front of the '.form'
    """
    if os.path.exists(target_file_name + '.{}'.format(form)):
        print("The path [{}] exists.".format(target_file_name + '.{}'.format(form)))
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        newpath = target_file_name + '-{}-v{}'.format(timestamp, np.random.randint(0, 1e8)) + '.{}'.format(form)
        print("Replace it by [{}].".format(newpath))
    else:
        newpath = target_file_name + '.{}'.format(form)
    return newpath

# functions for building loops enumerating all possible combinations of hyper-parameters
def load_loop_params(PARAMS, *args):
    numTotal = 1
    loopParams_dict = {}
    for key in args:
        if not isinstance(PARAMS[key], list):
            PARAMS[key] = [PARAMS[key]]
        loopParams_dict[key] = PARAMS[key]
        numTotal *= len(PARAMS[key])
        
    return loopParams_dict, numTotal

def pick_loop_params(idx, num_total, loop_params_dict):
    assert idx < num_total
    A = num_total
    params_list = []
    for value in loop_params_dict.values():
        length = len(value)
        A //= length
        params_list.append(value[(idx // A) % length])
    return params_list

def pick_loop_params_dict(idx, num_total, loop_params_dict):
    assert idx < num_total
    A = num_total
    params_dict = {}
    for key, value in loop_params_dict.items():
        length = len(value)
        A //= length
        params_dict[key] = value[(idx // A) % length]
    return params_dict