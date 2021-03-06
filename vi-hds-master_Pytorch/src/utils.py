import datetime
import os
import re
import shutil

import yaml

def get_data_directory():
    """
    Returns directory where observation datasets are stored (default: "data")
    """
    data_dir = os.getenv('INFERENCE_DATA_DIR')
    if data_dir:
        return data_dir
    else:
        return "data"

def get_results_directory():
    """
    Returns mount directory of remote machine on local, where inference results are to be stored (default: "results")
    """
    results_dir = os.getenv('INFERENCE_RESULTS_DIR')
    if results_dir:
        return results_dir
    else:
        return "results"

def load_config_file(filename):
    if filename is None:
        return None
    with open(filename, 'r') as stream:
        return yaml.unsafe_load(stream)

def default_get_value(dct, key, default_value, verbose=False):
    if key in dct:
        return dct[key]
    if verbose:
        print("%s using default %s" % (key, str(default_value)))
    return default_value

def apply_defaults(spec):
    params = {
        'solver': 'modeulerwhile',
        'use_laplace' : False,
        'n_filters' : 10,
        'filter_size' :  10,
        'pool_size' : 5,
        'lambda_l2' : 0.001,
        'lambda_l2_hidden' : 0.001,
        'n_hidden' : 50,
        'n_hidden_decoder' : 50,
        'n_batch' : 36,
        'data_format' : 'channels_last',
        'precision_type' : 'constant',
        'precision_alpha' : 1000.0,
        'precision_beta' : 1.0,
        'init_prec' : 0.00001,
        'init_latent_species' : 0.001,
        #'transfer_func' : tf.nn.tanh,
        'n_hidden_decoder_precisions' : 20,
        'n_growth_layers' : 4,
        'tb_gradients' : False,
        'plot_histograms' : False
    }
    for k in spec:
        params[k] = spec[k]
    return params

class Trainer(object):
    """Collection functions and attributes for training a Model"""
    def __init__(self, args, add_timestamp=False):
        self.results_dir = get_results_directory() #存放结果的目录
        self.experiment = args.experiment #实验名称
        self.yaml_file_name = args.yaml #参数数据存放的文件位置
        self.create_logging_dirs(add_timestamp) #实验结果的存放位置

    def _unique_dir_name(self, experiment, add_timestamp):
        now = datetime.datetime.now().isoformat()
        time_code = re.sub('[^A-Za-z0-9]+', '', now)  # current date and time concatenated into str for logging
        if add_timestamp is True:
            experiment += "_" + time_code
        return os.path.join(self.results_dir, experiment)

    def create_logging_dirs(self, add_timestamp=False):
        self.tb_log_dir = self._unique_dir_name(self.experiment, add_timestamp) #unammed_一系列数字字母
        os.makedirs(self.tb_log_dir, exist_ok=True)
        shutil.copyfile(self.yaml_file_name,
                        os.path.join(self.tb_log_dir, os.path.basename(self.yaml_file_name))) #把file1复制到file2中去