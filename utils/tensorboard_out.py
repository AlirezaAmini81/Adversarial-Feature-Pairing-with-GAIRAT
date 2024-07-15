import os
import shutil
import torch
import sys 

import torch
from tensorboardX import SummaryWriter
import random
import json 

# relative import hacks (sorry)
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) # for bash user
os.chdir(parentdir) # for pycharm user


def load_vars(filepath):
    """
    Loads variables from the given filepath.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filepath}' does not exist.")

    try:
        vs = torch.load(filepath)
        return vs
    except Exception as e:
        backup_filepath = '{}.old'.format(filepath)
        if os.path.exists(backup_filepath):
            shutil.copyfile(backup_filepath, filepath)
            os.remove(backup_filepath)
        raise e


def find_directories_with_file(directory, target_file_name):
    matching_directories = []

    for dirpath, dirnames, filenames in os.walk(directory):
        # Check if the target file is in the list of filenames
        if target_file_name in filenames:
            # If found, add the current directory to the list
            matching_directories.append(dirpath)

    return matching_directories


# directory = "/home/hamed/Storage/term9/LDA results/wj1"
directory = "/home/hamed/Storage/term9/LDA results"

target_file = "args.json"  # Replace with the name of the file you're looking for

def create_name(args):
    return str(args['model']) + str(args['loss']) + str(args['alpha'])

result = find_directories_with_file(directory, target_file)
if result:
    print("Directories containing '{}' file:".format(target_file))
    for dir_path in result:
        print(dir_path)
        with open(f'{dir_path}/args.json') as user_file:
            parsed_json = json.load(user_file)
            print(parsed_json)
            bs = parsed_json['batch_size']
            loss = parsed_json['loss']
            loaded_vars = load_vars(f"{dir_path}/losses.rar")
            print(loaded_vars.keys())

            writer = SummaryWriter(
                # log_dir=f'{dir_path}/tensor_board',
                log_dir=f'{dir_path}/tensor_board, bs: {bs}, loss: {loss}',
                # comment=f'{create_name(parsed_json)}'
            )

            for i, acc in enumerate(loaded_vars['test_acc']):
                writer.add_scalar('Accuracy/test', acc, i)
                writer.add_scalar('Accuracy/train', loaded_vars['train_acc'][i], i)
            
            writer.close()

else:   
    print("No directories containing '{}' file found.".format(target_file))
