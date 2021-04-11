from __future__ import absolute_import
from typing import Any, Dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import encoders
from src.decoders import ODEDecoder
import src.distributions as ds
from src.vi import GeneralLogImportanceWeights
from models.base_model import NeuralPrecisions


class Decoder(nn.Module):
    '''
    Decoder network
    '''

    def __init__(self, params: Dict[str, Any], procdata):
        super(Decoder, self).__init__()
        self.linear4 = nn.Linear(7, params['n_y']) #nn.Linear(7, self.n_y) No.4--》device_offsets
        n_in = 4 + params['n_latent_species'] + params['n_x'] + params['n_y'] + params['n_z'] + len(procdata.conditions) + procdata.device_depth
        n_out = params['n_hidden_decoder']
        inp = nn.Linear(n_in, n_out)  # activation = tf.nn.tanh output_size:50   nn.Linear(27,25)
        act_layer = nn.Linear(n_out, 4 + params['n_latent_species'])  # blackbox_growth #output_size:6   nn.Linear(25,6)
        deg_layer = nn.Linear(n_out, 4 + params['n_latent_species'])  # blackbox_degradation #output_size:6
        self.state_act = nn.Sequential(inp, nn.ReLU(), act_layer, nn.Sigmoid())
        self.state_deg = nn.Sequential(inp, nn.ReLU(), deg_layer, nn.Sigmoid())

        inputs = (4 + params['n_latent_species'] +4) + params['n_x'] + params['n_y'] + params['n_z'] + len(procdata.conditions) + procdata.device_depth
        self.neural_precisions = NeuralPrecisions(4, params['n_hidden_decoder_precisions'], inputs) #NeuralPrecisions(4,20,31)



    def forward(self, placeholders, encoder, times, params, device):
        ode_decoder = ODEDecoder(params)  # __init__函数起作用了
        # List(str), e.g. ['OD', 'RFP', 'YFP', 'CFP', 'F510', 'F430', 'LuxR', 'LasR']
        self.names = ode_decoder.ode_model.species  # list(str)
        # self.x_sample: Tensor of float32, shape e.g. (?, ?, ?, 8)
        # self.x_post_sample: Tensor of float32, shape e.g. (?, ?, ?, 4)
        # self.device_conditioned: Dict[str,Tensor], keys e.g. 'aS', 'aR'.
        self.x_sample, self.x_post_sample = ode_decoder( #__call__ function #x_sample:（batch_size,n_iwae,86,10），x_post_sample:(batch_size,n_iwae,86,4)
            placeholders['conds_obs'], placeholders['dev_1hot'], times, encoder.theta, encoder.clipped_theta, self.linear4, self.state_act, self.state_deg, self.neural_precisions, device,
            condition_on_device=True)
        return self

class Encoder(nn.Module):
    '''
    Encoder network
    '''

    def __init__(self, verbose: bool, parameters: 'Parameters', procdata):
        super(Encoder,self).__init__()
        ##Define Neural Network module
        self.encoder1 = encoders.ConditionalEncoder(parameters.params_dict)
        #Define the network for global-conditioned parameters
        if hasattr(parameters, "g_c"):
            Dict_of_global_cond_parameters = {}
            for i in range(1, len(parameters.g_c.list_of_params)+1): #range(1,3)
                mu = nn.Linear(procdata.device_depth, 1, bias=False)
                prec = nn.Linear(procdata.device_depth, 1, bias=False)
                Dict_of_global_cond_parameters['y%d_mu' % i] = mu
                Dict_of_global_cond_parameters['y%d_log_prec' % i] = prec
            self.global_cond_parameters = nn.ModuleDict(Dict_of_global_cond_parameters)
        #Define network for local parameters
        if hasattr(parameters, "l"):
            Dict_of_local_parameters = {}
            for i in range(1, len(parameters.l.list_of_params)+1):
                mu = nn.Linear(parameters.params_dict['n_hidden']+procdata.device_depth, 1, bias=True) #nn.Linear(57,1)
                prec = nn.Linear(parameters.params_dict['n_hidden']+procdata.device_depth, 1, bias=True)
                Dict_of_local_parameters['z%d_mu' % i] = mu
                Dict_of_local_parameters['z%d_log_prec' % i] = prec
            self.local_parameters = nn.ModuleDict(Dict_of_local_parameters)

        #Define learnable shift value for global parameters
        if hasattr(parameters, "g"):
            Dict_of_global_parameters = {} #x1-x5
            for i in range(1, len(parameters.g.list_of_params)+1):
                mu = nn.Parameter(torch.tensor(0.))
                prec = nn.Parameter(torch.tensor(1.))
                # list_of_x = nn.ParameterList([mu,prec])
                Dict_of_global_parameters['x%d_mu' % i] = mu
                Dict_of_global_parameters['x%d_log_prec' % i] = prec
            self.global_parameters = nn.ParameterDict(Dict_of_global_parameters)

    def forward(self, verbose, parameters, placeholders):
        # time-series of species differences: x_delta_obs is BATCH x (nTimes-1) x nSpecies
        x_delta_obs = placeholders['x_obs'][:, 1:, :] - placeholders['x_obs'][:, :-1,:]
        x_delta_obs = x_delta_obs.permute([0, 2, 1]).to(torch.float32)
        # ChainedDistribution
        self.q = self.set_up_q(verbose, parameters, placeholders, x_delta_obs, self.global_parameters, self.global_cond_parameters, self.local_parameters)
        # DotOperatorSamples
        self.theta = self.q.sample(placeholders['u'], verbose)  # return a dot operating theta
        # List of (about 30) strings
        #self.theta_names = self.theta.keys
        self.theta_names = self.q.get_theta_names()
        if verbose:
            print('THETA ~ Q')
            print(self.theta)
        # Tensor of float32. theta is in [batch, iwae_samples, theta_dim] iwae_samples-->importance weighted samples
        self.log_q_theta = self.q.log_prob(self.theta)  # initial log density # [batch, iwae_samples]
        # ChainedDistribution
        self.p = self.set_up_p(verbose, parameters)
        # DotOperatorSamples
        self.clipped_theta = self.p.clip(self.theta, stddevs=4)
        # Tensor of float
        self.log_p_theta = self.p.log_prob(self.theta)
        return self

    def set_up_p(self, verbose: bool, parameters: 'Parameters'):
        """Returns a ChainedDistribution"""
        p_vals = LocalAndGlobal(
            # prior: local: may have some dependencies in theta (in hierarchy, local, etc)
            ds.build_p_local(parameters, verbose, self.theta),
            ds.build_p_global_cond(parameters, verbose, self.theta),
            # prior: global should be fully defined in parameters
            ds.build_p_global(parameters, verbose, self.theta),
            ds.build_p_constant(parameters, verbose, self.theta))
        if verbose:
            p_vals.diagnostic_printout('P')
        return p_vals.concat("p")

    #@classmethod
    def set_up_q(self, verbose, parameters, placeholders, x_delta_obs, global_parameters, global_cond_parameters, local_parameters): #global_parameters is a Dict
        # Constants
        q_constant = ds.build_q_constant(parameters, verbose) #ds的意思：distributions
        # q: global, device-dependent distributions
        q_global_cond = ds.build_q_global_cond(parameters, placeholders['dev_1hot'], placeholders['conds_obs'], verbose, global_cond_parameters, plot_histograms=parameters.params_dict["plot_histograms"])
        # q: global, independent distributions
        q_global = ds.build_q_global(parameters, verbose, global_parameters)
        # q: local, based on amortized neural network
        if len(parameters.l.list_of_params) > 0:
            #encode = encoders.ConditionalEncoder(parameters.params_dict)
            approx_posterior_params = self.encoder1(x_delta_obs)
            q_local = ds.build_q_local(parameters, approx_posterior_params, placeholders['dev_1hot'], placeholders['conds_obs'], verbose, local_parameters)
                        #kernel_regularizer = tf.keras.regularizers.l2(0.01))
        else:
            q_local = ds.ChainedDistribution(name="q_local")
        q_vals = LocalAndGlobal(q_local, q_global_cond, q_global, q_constant)
        if verbose:
            q_vals.diagnostic_printout('Q')
        return q_vals.concat("q")


class SessionVariables:
    """Convenience class to hold the output of one of the Session.run calls used in training."""
    def __init__(self, seq):
        """seq: a sequence of 9 or 10 elements."""
        n = 9
        assert len(seq) == n or len(seq) == (n+1)
        (self.log_normalized_iws, self.normalized_iws, self.normalized_iws_reshape,
         self.x_post_sample, self.x_sample, self.elbo, self.precisions, self.theta_tensors, self.q_params) = seq[:n]
        self.summaries = seq[n] if len(seq) == (n+1) else None #self.summaries = None

    def as_list(self):
        result = [self.log_normalized_iws, self.normalized_iws, self.normalized_iws_reshape,
                  self.x_post_sample, self.x_sample, self.elbo, self.precisions, self.theta_tensors, self.q_params]
        if self.summaries is not None:
            result.append(self.summaries)
        return result


class LocalAndGlobal:
    """Convenience class to hold any tuple of local, global-conditional and global values."""

    def __init__(self, loc, glob_cond, glob, const):
        self.loc = loc
        self.glob_cond = glob_cond
        self.glob = glob
        self.const = const

    @classmethod
    def from_list(self, seq):
        return LocalAndGlobal(seq[0], seq[1], seq[2], seq[3])

    def to_list(self):
        return [self.loc, self.glob_cond, self.glob, self.const]

    def sum(self):
        return self.loc + self.glob_cond + self.glob + self.const

    def concat(self, name,):
        """Returns a concatenation of the items."""
        concatenated = ds.ChainedDistribution(name=name)
        for chained in self.to_list():
            for item_name, distribution in chained.distributions.items():
                concatenated.add_distribution(item_name, distribution, chained.slot_dependencies[item_name])
        return concatenated

    def diagnostic_printout(self, prefix):
        print('%s-LOCAL\n%s' % (prefix, self.loc))
        print('%s-GLOBAL-COND\n%s' % (prefix, self.glob_cond))
        print('%s-GLOBAL\n%s' % (prefix, self.glob))
        print('%s-CONSTANT\n%s' % (prefix, self.const))


class Objective(nn.Module):
    def __init__(self, parameters, params_dict, model, times, procdata, device, dreg = True, verbose = False):
        super(Objective, self).__init__()
        self.parameter = parameters #for Encoder
        self.params_dict = params_dict #for Decoder
        self.procdata = procdata
        self.device = device
        self.verbose = verbose
        self.times = times
        self.model = model
        self.dreg = dreg
        self.encoder = Encoder(self.verbose, self.parameter, self.procdata)
        self.decoder = Decoder(self.params_dict, self.procdata)

    def elbo(self, placeholders):
        self.encoder = self.encoder(self.verbose, self.parameter, placeholders)
        self.decoder = self.decoder(placeholders, self.encoder, self.times, self.params_dict, self.device)

        self.log_p_observations_by_species = self.model.log_prob_observations(self.decoder.x_post_sample, placeholders['x_obs'],
                                                                         self.encoder.theta, self.decoder.x_sample)
        self.log_p_observations = torch.sum(self.log_p_observations_by_species, 2)  # (batch_size,n_iwae,4)-->(batch_size,n_iwae)
        # let the model decide what precisions to use.
        # pylint:disable=fixme
        # TODO: will work for constant time precisions, but not for decayed. (get precisions after log_prob called)
        _log_precisions, self.precisions = self.model.expand_precisions_by_time(self.encoder.theta, self.decoder.x_post_sample,
                                                                           placeholders['x_obs'],
                                                                           self.decoder.x_sample)  #4 precisions for 4 output channel
        self.log_unnormalized_iws = GeneralLogImportanceWeights(
            self.log_p_observations, self.encoder.log_p_theta, self.encoder.log_q_theta, beta=1.0)  # (batch_size,n_iwae)

        logsumexp_log_unnormalized_iws = torch.logsumexp(self.log_unnormalized_iws, 1,
                                                         keepdim=True)  # (batch_size,n_iwae)-->(batch_size,1)
        self.log_normalized_iws = self.log_unnormalized_iws - logsumexp_log_unnormalized_iws  # 广播机制，归一化
        self.normalized_iws = torch.exp(self.log_normalized_iws)
        self.normalized_iws_reshape = self.normalized_iws.reshape(-1) # directly flatten (batch_size,n_iwae)-->(batch_size*n_iwae,)
        iwae_cost = -torch.mean(logsumexp_log_unnormalized_iws - torch.log(
            torch.tensor(self.log_p_observations.shape[1]).type(torch.FloatTensor)))
        self.elbos = -iwae_cost

        if not self.dreg:
            # [batch_size, num_iwae_samples]
            self.vae_cost = -torch.mean(self.log_unnormalized_iws) #why doubly negative?????
            # corresponds to `model_loss`
            return self.vae_cost  # turn a maximization problem in to a minimization problem

        else:
            normalized_weights =   F.softmax(self.log_unnormalized_iws, 1)
            sq_normalized_weights = normalized_weights**2  # [batch_size, num_iwae]
            stopped_log_q_theta = self.encoder.q.log_prob(self.encoder.theta, stop_grad=True)  # (batch_size,n_iwae)
            stopped_log_weights = GeneralLogImportanceWeights(self.log_p_observations, self.encoder.log_p_theta,
                                                              stopped_log_q_theta,
                                                              beta=1.0)
            neg_iwae_grad = torch.sum(sq_normalized_weights * stopped_log_weights, 1)  # [batch_size]
            iwae_grad = -torch.mean(neg_iwae_grad)
            return iwae_grad

class TrainingLogData:
    '''A convenience class of data collected for logging during training'''
    def __init__(self):
        self.training_elbo_list = []
        self.validation_elbo_list = []
        self.batch_feed_time = 0.0
        self.batch_train_time = 0.0
        self.total_train_time = 0.0
        self.total_test_time = 0.0
        self.n_test = 0
        self.max_val_elbo = -float('inf')

