import numpy as np

from src.solvers import modified_euler_integrate
from src.utils import default_get_value
from src.procdata import ProcData

import torch
import torch.nn as nn

def log_prob_laplace(x_obs, x_post_sample, log_precisions, precisions):
    log_p_x = torch.log(0.5) + log_precisions - precisions * torch.abs(x_post_sample - x_obs)
    return log_p_x

def log_prob_gaussian(x_obs, x_post_sample, log_precisions, precisions):
    # https://en.wikipedia.org/wiki/Normal_distribution
    log_p_x = -0.5 * torch.log(2.0 * torch.tensor(np.pi)) + 0.5 * log_precisions - 0.5 * precisions * ((x_post_sample - x_obs)**2)
    return log_p_x

def expand_constant_precisions(precision_list):
    # e.g.: precision_list = [theta.prec_x, theta.prec_fp, theta.prec_fp, theta.prec_fp ]
    precisions = torch.stack(precision_list, -1)
    log_precisions = torch.log(precisions)
    precisions = tf.expand_dims(precisions, 2)
    log_precisions = tf.expand_dims(log_precisions, 2)
    return log_precisions, precisions

class BaseModel(object):
    # We need an init_with_params method separate from the usual __init__, because the latter is
    # called automatically with no arguments by pyyaml on creation, and we need a way to feed
    # params (from elsewhere in the YAML structure) into it. It would really be better construct
    # it properly after the structure has been loaded.
    # pylint: disable=attribute-defined-outside-init
    def init_with_params(self, params, procdata : ProcData):
        self.params = params
        self.relevance = procdata.relevance_vectors
        self.default_devices = procdata.default_devices
        self.device_depth = procdata.device_depth
        self.n_treatments = len(procdata.conditions)
        self.use_laplace = default_get_value(self.params, 'use_laplace', False, verbose=True)
        self.precision_type = default_get_value(self.params, 'precision_type', 'constant', verbose=True)
        self.species = None
        self.nspecies = None
        #self.layers = []]

    def gen_reaction_equations(self, theta, conditions, dev_1hot, decoder_linear_4 = None, decoder_state_act = None, decoder_state_deg = None, condition_on_device=True, decoder_neural_precisions = None):
        raise NotImplementedError("TODO: write your gen_reaction_equations")

    def get_precision_list(self, theta):
        return [theta.prec_x, theta.prec_rfp, theta.prec_yfp, theta.prec_cfp]

    def initialize_state(self, theta, treatments):
        raise NotImplementedError("TODO: write your initialize_state")

    def simulate(self, theta, times, conditions, dev_1hot, solver, decoder_linear_4,  decoder_state_act, decoder_state_deg, decoder_neural_precisions, device, condition_on_device=True):
        init_state = self.initialize_state(theta, conditions, device)#tf.concat([x0, h0, prec0], axis=2) 4+2+4=10 (batch_size,n_iwae,10)
        d_states_d_t = self.gen_reaction_equations(theta, conditions, dev_1hot, decoder_linear_4, decoder_state_act, decoder_state_deg, decoder_neural_precisions, condition_on_device) #conditions(batch_size,2) (x_dot,prec_dot):(batch_size,n_iwae,10)
        solver = 'modeuler'
        if solver == 'modeuler': #??????????????????????????????
            # Evaluate ODEs using Modified-Euler
            #t_state, f_state = modified_euler_integrate(d_states_d_t, init_state, times)
            t_state, f_state = modified_euler_integrate(d_states_d_t, init_state, times)
            t_state_tr = t_state.permute(0,1,3,2) #[0, 1, 3, 2])
            f_state_tr = f_state.permute(0,1,3,2) #tf.transpose(f_state, [0, 1, 3, 2])
        else:
            raise NotImplementedError("Solver <%s> is not implemented" % solver)
        return t_state_tr, f_state_tr

    def observe(cls, x_sample, _theta):
        x_predict = [
            x_sample[:, :, :, 0],
            x_sample[:, :, :, 0] * x_sample[:, :, :, 1],
            x_sample[:, :, :, 0] * (x_sample[:, :, :, 2] + x_sample[:, :, :, 4]), #F530???F480??????hidden_latent_state(??????n_hidden_latent_state??????2)
            x_sample[:, :, :, 0] * (x_sample[:, :, :, 3] + x_sample[:, :, :, 5])]
        x_predict = torch.stack(x_predict, axis=-1)
        return x_predict

    def add_time_dimension(self, p, x):
        time_steps = x.shape[1]
        p = p.repeat([1, 1, time_steps, 1]) #torch.tensor.repeat()
        return p

    def expand_precisions_by_time(self, theta, _x_predict, x_obs, _x_sample):
        precision_list = self.get_precision_list(theta)
        log_prec, prec = self.expand_precisions(precision_list)
        log_prec = self.add_time_dimension(log_prec, x_obs)
        prec = self.add_time_dimension(prec, x_obs)
        if self.precision_type == "decayed":
            time_steps = x_obs.shape[1]
            lin_timesteps = tf.reshape(tf.linspace(1.0, time_steps.value, time_steps.value), [1, 1, time_steps, 1])
            prec = prec / lin_timesteps
            log_prec = log_prec - tf.log(lin_timesteps)
        return log_prec, prec

    def expand_precisions(cls, precision_list):
        return expand_constant_precisions(precision_list)

    def log_prob_observations(self, x_predict, x_obs, theta, x_sample):
        log_precisions, precisions = self.expand_precisions_by_time(theta, x_predict, x_obs, x_sample) #log_precisions, precisions (batch_size.n_iwae,86,4)
        # expand x_obs for the iw samples in x_post_sample
        x_obs_ = torch.unsqueeze(x_obs, 1) #(batch_size,86.4)-->(batch_size,1,86,4)
        lpfunc = log_prob_laplace if self.use_laplace else log_prob_gaussian #operater:??????
        log_prob = lpfunc(x_obs_, x_predict, log_precisions, precisions) #x_obs_????????????????????? ??????sample?????????????????????precision(deviation)
        # sum along the time and observed species axes
        #log_prob = tf.reduce_sum(log_prob, [2, 3])
        # sum along the time axis
        log_prob = torch.sum(log_prob, 2) #??????axis=2?????? (batch_size,n_iwae,86,4)-->(batch_size,n_iwae,4)
        return log_prob


class NeuralPrecisions(nn.Module):
    def __init__(self, nspecies, n_hidden_precisions, inputs = None): #hidden_activation = tf.nn.tanh):
        super(NeuralPrecisions, self).__init__()
        '''Initialize neural precisions layers'''
        self.nspecies = nspecies
        if inputs is None:
            inputs = self.nspecies+1
        inp = nn.Linear(inputs, n_hidden_precisions) #inp: input layer (batch_size,24)-->(batch_size,25)
        act_layer = nn.Linear(n_hidden_precisions, 4)#(batch_size,25)-->(batch_size,4)
        deg_layer = nn.Linear(n_hidden_precisions, 4)#(batch_size,25)-->(batch_size,4)
        self.act = nn.Sequential(inp, nn.ReLU(), act_layer, nn.Sigmoid())#(batch_size,24)-->(batch_size,4)
        self.deg = nn.Sequential(inp, nn.ReLU(), deg_layer, nn.Sigmoid())#(batch_size,24)-->(batch_size,4)

        #for layer in [inp, act_layer, deg_layer]: #?????????????????????????????????tensorboard?????????????????????
            #weights, bias = layer.weights
            #variable_summaries(weights, layer.name + "_kernel", False)
            #variable_summaries(bias, layer.name + "_bias", False)

    def __call__(self, t, state, n_batch, n_iwae):
        reshaped_state = tf.reshape(state[:,:,:-4], [n_batch*n_iwae, self.nspecies])
        reshaped_var_state = tf.reshape(state[:,:,-4:], [n_batch*n_iwae, 4])
        t_expanded = tf.tile( [[t]], [n_batch*n_iwae, 1] )
        ZZ_vrs = tf.concat( [ t_expanded, reshaped_state ], axis=1 )
        vrs = tf.reshape(self.act(ZZ_vrs) - self.deg(ZZ_vrs)*reshaped_var_state, [n_batch, n_iwae, 4])
        return vrs