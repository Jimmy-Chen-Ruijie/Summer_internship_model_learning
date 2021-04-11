import os
import numpy as np
from src import procdata
from src.utils import Trainer


class XvalMerge(object):

    def __init__(self, args, data_settings):
        self.separated_inputs = data_settings["separate_conditions"]  # True/False
        self.device_names = data_settings["devices"]  # 'Pcat_Y81C76','RS100S32_Y81C76','RS100S34_Y81C76','R33S32_Y81C76','R33S34_Y81C76','R33S175_Y81C76'
        self.conditions = data_settings["conditions"]  # "C6","C12"
        self.elbo = []
        self.elbo_list = []
        self.epoch = args.epochs
        self.name = args.experiment  # 实验的名称 run_xval中是 'unnamed'
        self.label = args.experiment
        self.log_normalized_iws = []  # important weights
        self.precisions = []
        self.q_names = []
        self.q_values = []
        self.splits = []
        self.theta = []
        self.X_post_sample = []
        self.X_sample = []
        # from data_pair.val
        self.data_ids = []
        self.devices = []
        self.treatments = []
        self.trainer = trainer = Trainer(args, add_timestamp=True)
        self.X_obs = []
        # Attributes initialized elsewhere
        self.chunk_sizes = None
        self.ids = None
        self.names = None
        self.times = None
        self.xval_writer = None

    def add(self, split_idx, data_pair, val_results):
        if split_idx == 1:
            self.q_names = val_results["q_names"]
            self.names = val_results["names"]
            self.times = val_results["times"]
        self.elbo.append(val_results["elbo"])  # 都是列表 list,可用用于np.concatenate
        self.elbo_list.append(val_results["elbo_list"])
        self.log_normalized_iws.append(val_results["log_normalized_iws"])
        self.precisions.append(val_results["precisions"])
        self.q_values.append(val_results["q_values"])
        self.splits.append(split_idx)
        self.theta.append(val_results["theta"])
        self.X_post_sample.append(val_results["x_post_sample"])
        self.X_sample.append(val_results["x_sample"])

        self.data_ids.append(data_pair.val.original_data_ids)
        self.devices.append(data_pair.val.devices)
        self.treatments.append(data_pair.val.treatments)
        self.X_obs.append(data_pair.val.X)

    def finalize(self):
        print('Preparing cross-validation results')
        self.elbo = np.array([self.elbo[0].cpu()])  # 都转变为np.ndarray就是为了能够用numpy内置函数来保存数据
        self.elbo_list = np.array([[col.cpu() for col in row] for row in self.elbo_list])
        self.log_normalized_iws = np.concatenate([self.log_normalized_iws[0].cpu()], 0)
        self.precisions = np.concatenate([self.precisions[0].cpu()], 0)
        # self.q_values = [np.hstack(q) for q in np.array(self.q_values).transpose()]
        # self.q_values = np.hstack(self.q_values)
        #self.q_values = [np.concatenate([np.array(q[.cpu(), ndmin=1) for q in self.q_values[0]]) for i, _ in
         #                enumerate(self.q_names)]
        self.X_post_sample = np.concatenate([self.X_post_sample[0].cpu()], 0)
        self.X_sample = np.concatenate([self.X_sample[0].cpu()], 0)

        self.devices = np.concatenate(self.devices, 0)
        self.treatments = np.concatenate(self.treatments, 0)
        self.X_obs = np.concatenate(self.X_obs, 0)

        self.chunk_sizes = np.array([len(ids) for ids in self.data_ids])
        self.ids = np.hstack(self.data_ids)

    def save(self):
        location = self.trainer.tb_log_dir

        def save(base, data):
            np.save(os.path.join(location, base), data)

        def savetxt(base, data):
            np.savetxt(os.path.join(location, base), np.array(data, dtype=str), delimiter=" ", fmt="%s")

        print("Saving to: %s" % location)
        save("xval_result_elbo", self.elbo)
        save("xval_result_elbo_list", self.elbo_list)
        save("xval_result_log_normalized_iws", self.log_normalized_iws)
        save("xval_result_precisions", self.precisions)
        savetxt("xval_result_q_names.txt", self.q_names)
        #save("xval_result_q_values", self.q_values)
        save("xval_result_theta", self.theta)
        save("xval_result_X_post_sample", self.X_post_sample)
        save("xval_result_X_sample", self.X_sample)

        savetxt("xval_result_device_names.txt", self.device_names)
        save("xval_result_devices", self.devices)
        save("xval_result_treatments", self.treatments)
        save("xval_result_X_obs", self.X_obs)

        save("xval_result_chunk_sizes", self.chunk_sizes)
        save("xval_result_ids", self.ids)
        savetxt("xval_result_names.txt", self.names)
        save("xval_result_times", self.times)