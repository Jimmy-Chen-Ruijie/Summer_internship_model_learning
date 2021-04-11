from __future__ import absolute_import
import argparse
import os
import time
from typing import Any, Dict, List

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src import procdata
from src.parameters import Parameters
from src.convenience import LocalAndGlobal, Objective, Decoder, Encoder, TrainingLogData, SessionVariables
from src.xval import XvalMerge
from src import utils

from models.base_model import NeuralPrecisions

class Runner:
    """A class to set up, train and evaluate a variation CRN model, holding out one fold
    of a data set for validation. See "run_on_split" below for how to set it up and run it."""

    def __init__(self, args, split=None, trainer=None):
        """
        :param args: a Namespace, from argparse.parse_args
        :param split: an integer between 1 and args.folds inclusive, or None
        :param trainer: a Trainer instance, or None
        """
        self.procdata = None
        # Command-line arguments (Namespace)
        self.args = self._tidy_args(args, split)
        # TODO(dacart): introduce a switch to allow non-GPU use, achieved with:
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # Utility methods for training a model
        self.trainer = trainer or utils.Trainer(args, add_timestamp=True)
        # Attributes set in other methods:
        # Conditions, input signals of the data that are being modelled
        self.conditions = None
        # DatasetPair, from a training Dataset and validation Dataset
        self.dataset_pair = None
        # Decoder and encoder networks
        self.decoder = None
        self.encoder = None
        # Number of instances in a batch (int)
        self.n_batch = None
        # Number of "theta" parameters: local, global-conditional and global (int)
        self.n_theta = None
        # Collection of attributes related to training objective
        self.objective = None
        # Value of spec["params"] from YAML file (dict)
        self.params_dict = None
        # Collection of placeholder attributes, each a Tensor, fed with new values for each batch
        self.placeholders = None
        # Training feed_dict: dict from placeholder Tensor to np.array
        self.train_feed_dict = None
        # TrainingStepper object
        self.training_stepper = None
        # Validation feed_dict keys: dict from placeholder Tensor to np.array
        self.val_feed_dict = None
        # Model path for storing best weights so far
        self.model_path = os.path.join(self.trainer.tb_log_dir, 'saver', 'sess_max_elbo')

    def _tidy_args(cls, args, split):
        print("Called run_xval with ")
        print(args)
        if getattr(args, 'heldout', None):
            print("Heldout device is %s" % args.heldout)
        else:
            args.heldout = None # in case not defined at all
            if split is not None:
                args.split = split
            print("split = %d" % args.split)
        if args.epochs < args.test_epoch:
            msg = "No test epochs possible with epochs = %d and test_epochs = %d"%(args.epochs,args.test_epoch)
            raise Exception(msg)
        return args

    @staticmethod
    def _decide_indices(size, randomize):
        if randomize:
            return np.random.permutation(size)
        return np.arange(size, dtype=int)

    def _init_chunks(self):
        size = self.dataset_pair.n_train #234
        chunk_size = self.n_batch #36
        indices = self._decide_indices(size, randomize=True) #随机打乱返回一个打乱的，长度为234的np.ndarray
        return [indices[i:i + chunk_size] for i in range(0, size, chunk_size)] #从0数到234，步长为36，返回一个列表，该列表中用7个np.ndarray

    def _init_fold_chunks(cls, size, folds, randomize):
        indices = cls._decide_indices(size, randomize)
        return np.array_split(indices, folds)

    @staticmethod
    def _decide_dataset_pair(all_ids, val_ids, loaded_data, data_settings):
        train_ids = np.setdiff1d(all_ids, val_ids) #在all_ids，但不在val_ids的已排序的唯一值
        train_data, val_data = procdata.split_by_train_val_ids(loaded_data, train_ids, val_ids)
        return procdata.DatasetPair(procdata.Dataset(data_settings, train_data, train_ids), procdata.Dataset(data_settings, val_data, val_ids))

    def _prepare_data(self, data_settings: Dict[str, str]):
        '''data: a dictionary of the form {'devices': [...], 'files': [...]}
              where the devices are names like Pcat_Y81C76 and the files are csv basenames.
        Sets self.dataset_pair to hold the training and evaluation datasets.'''
        # "Make a usable dataset (notions of repeats and conditions)"
        # Establish where the csv files in data['files'] are to be found.
        data_dir = utils.get_data_directory() #返回存放data的文件夹名称（csv格式）
        # Map data arguments into internal structures
        #self.conditions = data_settings["conditions"]
        # Load all the data in all the files, merging appropriately (see procdata for details).
        self.procdata = procdata.ProcData(data_settings)
        loaded_data = self.procdata.load_all(data_dir)
        np.random.seed(self.args.seed)
        if self.args.heldout: #None (Not None = True)
            # We specified a holdout device to act as the validation set.
            d_train, d_val, train_ids, val_ids = procdata.split_holdout_device(self.procdata, loaded_data, self.args.heldout)
            train_dataset = procdata.Dataset(data_settings, d_train, train_ids)
            val_dataset = procdata.Dataset(data_settings, d_val, val_ids)
            self.dataset_pair = procdata.DatasetPair(train_dataset, val_dataset)
        else:
            # Number of conditions (wells) in the data.
            loaded_data_length = loaded_data['X'].shape[0] # len(loaded_data['X']) #312个数据 （312 differently conditioned samples）
            # The validation set is determined by the values of "split" and args.folds.
            # A list of self.args.folds (roughly) equal size numpy arrays of int, total size W, values in [0, W-1]
            # where W is the well (condition) count, all distinct.
            val_chunks = self._init_fold_chunks(loaded_data_length, self.args.folds, randomize=True)
            assert len(val_chunks) == self.args.folds, "Bad chunks"
            # All the ids from 0 to W-1 inclusive, in order.
            all_ids = np.arange(loaded_data_length, dtype=int) # 0 - 311 总共312个数字
            # split runs from 1 to args.folds, so the index we need is one less.
            # val_ids is the indices of data items to be used as validation data.
            val_ids = np.sort(val_chunks[self.args.split - 1])
            # A DatasetPair object: two Datasets (one train, one val) plus associated information.
            self.dataset_pair = self._decide_dataset_pair(all_ids, val_ids, loaded_data, data_settings)

    def _random_noise(self, size):
        return np.random.randn(size, self.args.test_samples, self.n_theta)

    def _create_feed_dicts(self):
        self.train_feed_dict = self.dataset_pair.train.create_feed_dict(self._random_noise(self.dataset_pair.n_train), self.device) #GPU out of memory...
        self.val_feed_dict = self.dataset_pair.val.create_feed_dict(self._random_noise(self.dataset_pair.n_val), self.device)
        #self.val_feed_dict = self.dataset_pair.val.create_feed_dict(self._random_noise(9))

    def set_up(self, data_settings, params):
        self.params_dict = params  # object Runner的method: set_up

        # time some things, like epoch time
        start_time = time.time()

        # ---------------------------------------- #
        #     DEFINE XVAL DATASETS                 #
        # ---------------------------------------- #

        # Create self.dataset_pair: DatasetPair containing train and val Datasets.
        self._prepare_data(data_settings)
        # Number of instances to put in a training batch.
        self.n_batch = min(self.params_dict['n_batch'],
                           self.dataset_pair.n_train)  # 如果训练集样本个数小于一个batch中的样本个数的话 就整个训练样本一起训练了

        # This is already a model object because of the use of "!!python/object:... in the yaml file.
        model = self.params_dict["model"]
        # Set various attributes of the model
        model.init_with_params(self.params_dict, self.procdata)

        # Import priors from YAML
        parameters = Parameters()  # instantiate a class
        parameters.load(self.params_dict)

        print("----------------------------------------------")
        if self.args.verbose:
            print("parameters:")
            parameters.pretty_print()
        n_vals = LocalAndGlobal.from_list(
            parameters.get_parameter_counts())  # (10,0,0,4), 有14个属性，分别记录着constant,global,global_conditioned,local parameters的维度
        self.n_theta = n_vals.sum()  # 所有parameters的个数 总共14个parameters

        #     Pytorch PARTS        #

        # DEFINE THE OBJECTIVE
        print("Set up model")
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #device = torch.device('cpu')

        self.device = device
        # create feed dictionaries for training dataset and validation dataset respectively
        self._create_feed_dicts()

        self.objective = Objective(parameters, self.params_dict, model, self.dataset_pair.times, self.procdata, self.device, self.args.dreg )#, self.args.verbose)
        # Initialize the weight of neural networks
        self.weight_orthogonal_initialization()
        # feed the model to the GPU (if GPU is available)
        self.objective.to(device)

        self.optimizer = torch.optim.Adam(self.objective.parameters(), lr=5e-3)

    def _create_session_variables(self):
        return SessionVariables([
            self.objective.log_normalized_iws, #Yes
            self.objective.normalized_iws, #Yes
            self.objective.normalized_iws_reshape,#Yes
            self.objective.decoder.x_post_sample, #Yes
            self.objective.decoder.x_sample, #Yes
            self.objective.elbos, #Yes
            self.objective.precisions, #Yes
            self.objective.encoder.theta.get_tensors(), #Yes
            self.objective.encoder.q.get_tensors()]) #Yes

    def _run_batch(self, beta_val, i_batch):
        # random indices of training data for this batch
        # i_batch = np.random.permutation(n_train)[:n_batch]
        # placeholders.u: standard normals driving everything!
        u_value = np.random.randn(len(i_batch), self.args.train_samples, self.n_theta) #返回一个（36,200,14） (batch_size,n_iwae,14)
        batch_feed_dict = self.dataset_pair.train.create_feed_dict_for_index(i_batch, beta_val, u_value, self.device) #把csv中具体的数值传进来
        beta_val = torch.Tensor([np.minimum(1.0, beta_val * 1.01)])
        # keep track of this time, sometimes (not here) this can be inefficient
        # take a training step
        self.optimizer.zero_grad()
        # get the regularization term w.r.t weights of certain layer
        regularization = self.get_regularization()
        loss = self.objective.elbo(batch_feed_dict) + regularization
        loss.backward()
        self.optimizer.step()
        # Apply hard constraint on the bias of act/ deg layer defined in the Neuralprecision of Decoder
        self.Neural_precision_bias_nonneg()

        return beta_val

    def Neural_precision_bias_nonneg(self):
        for child_of_objective in self.objective.children():
            if isinstance(child_of_objective, Decoder):
                # print(child)
                for child_of_decoder in child_of_objective.children():
                    if isinstance(child_of_decoder, NeuralPrecisions):
                        Sequential_list_act = [child for child in child_of_decoder.act.children()]
                        Sequential_list_act[2].bias.data.clamp_(0, float('inf'))
                        Sequential_list_deg = [child for child in child_of_decoder.deg.children()]
                        Sequential_list_deg[2].bias.data.clamp_(0, float('inf'))

    def get_regularization(self):
        regularization = 0
        for child_of_objective in self.objective.children():
            if isinstance(child_of_objective, Encoder):
                # print(child_of_objective.local_parameters)
                for child in child_of_objective.local_parameters.children():
                    # print(child.weight)
                    regularization += 0.01*torch.norm(child.weight, 2)
                for child in child_of_objective.encoder1.children():
                    # print(child)
                    if hasattr(child, 'weight'):
                        regularization += 0.1*torch.norm(child.weight, 2)
        return regularization

    def weight_orthogonal_initialization(self):
        for child_of_objective in self.objective.children():
            if isinstance(child_of_objective, Encoder):
                for child in child_of_objective.encoder1.children():
                    # print(child)
                    if hasattr(child, 'weight'):
                        nn.init.orthogonal_(child.weight)

    def ELBO_to_tensorboard(self, writer, epoch):
        # write a serie of elbos into tensorboard
        writer.add_scalar('ELBO/elbo', self.objective.elbos, epoch)
        log_p = torch.mean(torch.logsumexp(self.objective.log_p_observations, 1))
        writer.add_scalar('ELBO/log_p', log_p, epoch)
        for i, plot in enumerate(self.procdata.signals):
            log_p_by_species = torch.mean(torch.logsumexp(self.objective.log_p_observations_by_species[:, :, i], 1))
            writer.add_scalar('ELBO/log_p_' + plot, log_p_by_species, epoch)
        # Priors
        logsumexp_log_p_theta = torch.logsumexp(self.objective.encoder.log_p_theta, 1)
        writer.add_scalar('ELBO/log_prior', torch.mean(logsumexp_log_p_theta), epoch)
        logsumexp_log_q_theta = torch.logsumexp(self.objective.encoder.log_q_theta, 1)
        writer.add_scalar('ELBO/log_q', torch.mean(logsumexp_log_q_theta), epoch)


    def _run_session(self):

        held_out_name = self.args.heldout or '%d_of_%d' % (self.args.split, self.args.folds) #1_of_4
        # to write variable that we are interested in into tensorboard
        train_writer = SummaryWriter(os.path.join(self.trainer.tb_log_dir, 'train_%s' % held_out_name))
        valid_writer = SummaryWriter(os.path.join(self.trainer.tb_log_dir, 'valid_%s' % held_out_name))

        print("----------------------------------------------")
        print("Starting Session...")
        beta = 1.0

        log_data = TrainingLogData()
        print("===========================")
        if self.args.heldout:
            split_name = 'heldout device = %s' % self.args.heldout
        else:
            split_name = 'split %d of %d' % (self.args.split, self.args.folds)
        print("Training: %s"%split_name) #Training: split 1 of 4

        # start training process
        for epoch in range(1, self.args.epochs + 1):
            epoch_start = time.time()
            self.objective.train()
            epoch_batch_chunks = self._init_chunks()
            for i_batch in epoch_batch_chunks:
               beta = self._run_batch(beta, i_batch)
            log_data.total_train_time += time.time() - epoch_start

            if np.mod(epoch, 5) == 0:
            #if np.mod(epoch, self.args.test_epoch) == 0: #每20个epoch就打印一次：总共1000个epoch,所以打印50次
                self.objective.eval()
                with torch.no_grad():
                    print("epoch %4d" % epoch, end='', flush=True)
                    log_data.n_test += 1
                    test_start = time.time()
                    plot = np.mod(epoch, self.args.plot_epoch) == 0

                    #Evaluation on Training Dataset
                    self.train_feed_dict['u'] = torch.Tensor(np.random.randn(
                        self.dataset_pair.n_train, self.args.train_samples, self.n_theta)).to(self.device)  # (234,200,14) torch.Tensor(u_value).to(device)
                    self.objective.elbo(self.train_feed_dict)
                    training_output = self._create_session_variables()
                    #train_writer.add_scalar('elbo', self.objective.elbos, epoch)
                    self.ELBO_to_tensorboard(train_writer, epoch)
                    print(" | train (iwae-elbo = %0.4f, time = %0.2f, total = %0.2f)"%(training_output.elbo, log_data.total_train_time / epoch, log_data.total_train_time), end=' ', flush=True)
                    train_writer.flush()

                    #Evaluation on Validation Dataset
                    self.val_feed_dict['u'] = torch.Tensor(np.random.randn(
                        self.dataset_pair.n_val, self.args.test_samples, self.n_theta)).to(self.device)  # (36,1000,14) torch.Tensor(u_value).to(device)
                    self.objective.elbo(self.val_feed_dict)
                    validation_output = self._create_session_variables()
                    #valid_writer.add_scalar('elbo', self.objective.elbos, epoch)
                    log_data.total_test_time += time.time() - test_start
                    self.ELBO_to_tensorboard(valid_writer, epoch)
                    print(" | val (iwae-elbo = %0.4f, time = %0.2f, total = %0.2f)"%(validation_output.elbo, log_data.total_test_time / log_data.n_test, log_data.total_test_time))
                    valid_writer.flush()

                    # record the parameters of the best model
                    if validation_output.elbo > log_data.max_val_elbo:
                        log_data.max_val_elbo = validation_output.elbo
                        torch.save(self.objective, os.path.join(self.trainer.tb_log_dir, 'model_parameters'))

                    log_data.training_elbo_list.append(training_output.elbo)
                    log_data.validation_elbo_list.append(validation_output.elbo)

        train_writer.close()
        valid_writer.close()
        print("===========================")

        # load the parameters of the best model
        self.objective = torch.load(os.path.join(self.trainer.tb_log_dir, 'model_parameters'), map_location=torch.device('cuda'))
        # output the elbo of validation dataset
        with torch.no_grad():
            self.objective.elbo(self.val_feed_dict)
            validation_output = self._create_session_variables()
        return log_data, validation_output

    def run(self):
        log_data, validation_output = self._run_session()

        val_results = {"names": self.objective.decoder.names,
                       "times": self.dataset_pair.times,
                       "x_sample": validation_output.x_sample,
                       "x_post_sample": validation_output.x_post_sample,
                       "precisions": validation_output.precisions,
                       "log_normalized_iws": validation_output.log_normalized_iws,
                       "elbo": validation_output.elbo,
                       "elbo_list": log_data.validation_elbo_list,
                       "theta": validation_output.theta_tensors,
                       "q_names": self.objective.encoder.q.get_tensor_names(),
                       "q_values": validation_output.q_params}

        return self.dataset_pair, val_results



def create_parser(with_split: bool):
    parser = argparse.ArgumentParser(description='VI-HDS')
    #parser.add_argument('yaml', type=str, help='Name of yaml spec file')
    parser.add_argument('--yaml', type=str,default='../specs/dr_blackbox_xval_hierarchical.yaml', help='Name of yaml spec file')
    parser.add_argument('--experiment', type=str, default='unnamed', help='Name for experiment, also location of tensorboard and saved results')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs') #used to be 1000
    parser.add_argument('--test_epoch', type=int, default=20, help='Frequency of calling test') #used to be 20
    parser.add_argument('--plot_epoch', type=int, default=100, help='Frequency of plotting figures')
    parser.add_argument('--train_samples', type=int, default=200, help='Number of samples from q, per datapoint, during training')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of samples from q, per datapoint, during testing')
    parser.add_argument('--dreg', type=bool, default=True, help='Use DReG estimator') #If true, then use dreg (doubly reparametrized gradient estimator)
    parser.add_argument('--verbose', action='store_true', default=True, help='Print more information about parameter setup')
    if with_split:
        # We make --heldout (heldout device) and --split mutually exclusive. Really we'd like to say it's allowed
        # to specify *either* --heldout *or* --split and/or --folds, but argparse isn't expressive enough for that.
        # So if the user specifies --heldout and --folds, there won't be a complaint here, but --folds will be ignored.
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--heldout', type=str, help='name of held-out device, e.g. R33S32_Y81C76')
        group.add_argument('--split', type=int, default=1, help='Specify split in 1:folds for cross-validation')
    parser.add_argument('--folds', type=int, default=4, help='Cross-validation folds') #used to be 4
    return parser

def add_tensor_todevice(feed_dict,device):
    for k, v in feed_dict:
        feed_dict[k] = v.to(device)
    return feed_dict

def run_on_split(args, data_settings, para_settings, split=None, trainer=None):
    runner = Runner(args, split, trainer)
    runner.set_up(data_settings, para_settings)
    return runner.run()

def main():
    parser = create_parser(True)
    args = parser.parse_args()
    spec = utils.load_config_file(args.yaml) #spec is a dict of dicts of dicts
    data_settings = procdata.apply_defaults(spec["data"])
    para_settings = utils.apply_defaults(spec["params"])
    xval_merge = XvalMerge(args, data_settings)
    data_pair, val_results = run_on_split(args, data_settings, para_settings, split=None, trainer=xval_merge.trainer)
    xval_merge.add(1, data_pair, val_results)
    xval_merge.finalize()
    xval_merge.save()

if __name__ == "__main__":
    main()