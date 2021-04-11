from models.base_model import BaseModel, NeuralPrecisions
from src.utils import default_get_value
import torch
import torch.nn as nn

class DR_Blackbox( BaseModel ):

    def init_with_params( self, params, procdata ):
        super(DR_Blackbox, self).init_with_params( params, procdata )
        self.species = ['OD', 'RFP', 'YFP', 'CFP']
        self.nspecies = 4
        # do the other inits now
        self.n_z = params['n_z']
        self.n_hidden = params['n_hidden_decoder']
        self.n_latent_species = params['n_latent_species']
        self.init_latent_species = default_get_value(params, 'init_latent_species', 0.001)

    def observe( self, x_sample, theta ):
        #x0 = [theta.x0, theta.rfp0, theta.yfp0, theta.cfp0] #OD-x0 RFP-x0*x1 YFP-x0*x2 CFP-x0*x3
        x_predict = [ x_sample[:,:,:,0], \
                x_sample[:,:,:,0]*x_sample[:,:,:,1], \
                x_sample[:,:,:,0]*x_sample[:,:,:,2], \
                x_sample[:,:,:,0]*x_sample[:,:,:,3]]
        x_predict = torch.stack( x_predict, -1 ) #在最后一维叠加，因为list x_predict中有4个元素，所以是（?,?,?,4）
        return x_predict

class DR_BlackboxPrecisions( DR_Blackbox ):
    def init_with_params( self, params, procdata ):
        super(DR_BlackboxPrecisions, self).init_with_params( params, procdata ) #先初始化DR_Blackbox类并且运行初始化函数init_with_params,然后把类的属性继承给DR_BlackboxPrecisions上
        self.init_prec = params['init_prec']
        self.n_hidden_precisions = params['n_hidden_decoder_precisions'] #hidden precision的个数
        self.n_states = 4 + self.n_latent_species + 4

    def initialize_state(self, theta, _treatments, device):
        n_batch = theta.get_n_batch()
        n_iwae = theta.get_n_samples()
        zero = torch.zeros([n_batch, n_iwae])
        x0 = torch.stack([theta.init_x, theta.init_rfp, theta.init_yfp, theta.init_cfp], 2)
        h0 = torch.Tensor(n_batch, n_iwae, self.n_latent_species).to(device)
        h0.fill_(self.init_latent_species)
        prec0 = torch.Tensor(n_batch, n_iwae, 4).to(device)
        prec0.fill_(self.init_prec)
        #h0 = tf.fill([n_batch, n_iwae, self.n_latent_species], self.init_latent_species) #(36,1000)? (36,100)? self.n_latent_species=2
        #prec0 = tf.fill([n_batch, n_iwae, 4], self.init_prec) #
        return torch.cat([x0, h0, prec0], 2)

    def expand_precisions_by_time(self, theta, x_predict, x_obs, x_sample):
        var = x_sample[:, :, :, -4:]
        prec = 1.0 / var  # var: variation
        log_prec = torch.log(prec)
        return log_prec, prec

    def initialize_neural_states(self, n):
        '''Neural states'''
        inp = nn.Linear(n, self.n_hidden)  # activation = tf.nn.tanh output_size:50
        act_layer = nn.Linear(self.n_hidden, 4 + self.n_latent_species)  # blackbox_growth #output_size:6
        deg_layer = nn.Linear(self.n_hidden, 4 + self.n_latent_species)  # blackbox_degradation #output_size:6
        act = nn.Sequential(inp, nn.ReLU(), act_layer, nn.Sigmoid())
        deg = nn.Sequential(inp, nn.ReLU(), deg_layer, nn.Sigmoid())
        #for layer in [inp, act_layer, deg_layer]:  # 写自己version代码的时候暂时可以不用
            #weights, bias = layer.weights
            #variable_summaries(weights, layer.name + "_kernel", False)
            #variable_summaries(bias, layer.name + "_bias", False)
        return act, deg

    def gen_reaction_equations(self, theta: object, treatments: object, dev_1hot: object, decoder_linear_4 = None, decoder_state_act = None, decoder_state_deg = None, decoder_neural_precisions = None,
                               condition_on_device: object = True) -> object:
        n_iwae = theta.get_n_samples()
        n_batch = theta.get_n_batch()
        devices = dev_1hot.repeat([n_iwae,1])  # (batch_size*n_iwae,1) dev_1hot.repeat([n_iwae,1])
        treatments_rep = treatments.repeat([n_iwae,1])  # (batch_size*n_iwae,2)

        Z = []
        for i in range(1, self.n_z + 1):
            Z.append(getattr(theta, "z%d" % i))
        Z = torch.stack(Z, 2)  # (batch_size,n_iwae,5) theta.z1到theta.z5的堆砌

        n = 4 + self.n_latent_species + self.n_z + self.n_treatments + self.device_depth
        #states_act, states_deg = self.initialize_neural_states(n)  # 两个都是tensorflow神经网络对象，都为两层全连接层神经网络，（batch_size,20）-->(batch_size,6),这两个在reaction_equation函数中起到作用？
        states_act = decoder_state_act
        states_deg = decoder_state_deg
        #neural_precisions = NeuralPrecisions(self.nspecies, self.n_hidden_precisions,
         #                                    # self.n_hidden_precisions=25, self.nspecies=4, self.act;self.deg反而起作用了
          #                                   inputs=self.n_states + self.n_z + self.n_treatments + self.device_depth,
                                             # self.n_states=10, inputs=24
          #                                   hidden_activation=tf.nn.relu)
        neural_precisions = decoder_neural_precisions

        def reaction_equations(state, t):
            all_reshaped_state = torch.reshape(state, [n_batch * n_iwae,
                                                    self.n_states])  # (batch_size,n_iwae,10)-->(batch_size*n_iwae,10)

            # split for precisions and states
            reshaped_state = all_reshaped_state[:, :-4]  # states 6 dimensional
            reshaped_var_state = all_reshaped_state[:, -4:]  # precisions 4 dimensional

            ZZ_states = torch.cat([reshaped_state, \
                                   torch.reshape(Z, [n_batch * n_iwae, self.n_z]), \
                                   treatments_rep, \
                                   devices], 1)  # (batch_size*n_iwae,6+5+2+7)=(batch_size*n_iwae,20)
            states = states_act(ZZ_states) - states_deg(ZZ_states) * reshaped_state  # states is a 6 dimensional tensor
            ZZ_vrs = torch.cat([all_reshaped_state, \
                                torch.reshape(Z, [n_batch * n_iwae, self.n_z]), \
                                treatments_rep, \
                                devices], 1)  # (batch_size*n_iwae,10+5+2+7)=(batch_size*n_iwae,24)  for x+h (state+hidden_state)
            vrs = neural_precisions.act(ZZ_vrs) - neural_precisions.deg(
                ZZ_vrs) * reshaped_var_state  # vrs is a 4 dimensinal vector

            return torch.reshape(torch.cat([states, vrs], 1),
                              [n_batch, n_iwae, self.n_states])  # reaction_equations的返回值 (batch_size, n_iwae,10)

        return reaction_equations  # gen_reaction_equations的返回值


class DR_HierarchicalBlackbox(DR_BlackboxPrecisions):

    def init_with_params(self, params, procdata):
        super(DR_HierarchicalBlackbox, self).init_with_params(params, procdata)
        # do the other inits now
        self.n_x = params['n_x']
        self.n_y = params['n_y']
        self.n_z = params['n_z']
        self.n_latent_species = params['n_latent_species']
        self.n_hidden_species = params['n_hidden_decoder']
        self.n_hidden_precisions = params['n_hidden_decoder_precisions']
        self.init_latent_species = default_get_value(params, 'init_latent_species', 0.001)
        self.init_prec = default_get_value(params, 'init_prec', 0.00001)

    def gen_reaction_equations(self, theta, treatments, dev_1hot, decoder_linear_4 = None, decoder_state_act = None, decoder_state_deg = None, decoder_neural_precisions = None, condition_on_device=True):
        n_iwae = torch.tensor(theta.z1.shape[1])
        n_batch = torch.tensor(theta.z1.shape[0])
        devices = dev_1hot.repeat([n_iwae,1])
        treatments_rep = treatments.repeat([n_iwae,1])
        #devices = tf.tile(dev_1hot, [n_iwae, 1])
        #treatments_rep = tf.tile(treatments, [n_iwae, 1])

        # locals
        Z = []
        if self.n_z > 0:
            for i in range(1, self.n_z + 1):
                Z.append(getattr(theta, "z%d" % i))
            Z = torch.stack(Z, 2) #(batch_size,n_iwae,self.n_z)

        # global conditionals
        Y = []
        if self.n_y > 0:
            for i in range(1, self.n_y + 1):
                nm = "y%d" % i
                Y.append(getattr(theta, nm))
            Y = torch.stack(Y, 2)
            Y_reshaped = torch.reshape(Y, [n_batch * n_iwae, self.n_y])
            #offset_layer = nn.Linear(7, self.n_y) #根据dev_1hot的维度来的，后续要把7泛化
            offset_layer = decoder_linear_4
            Y_reshaped = Y_reshaped + offset_layer(devices)
            Y = torch.reshape(Y_reshaped, [n_batch, n_iwae, self.n_y])

        # globals
        X = []
        if self.n_x > 0:
            for i in range(1, self.n_x + 1):
                X.append(getattr(theta, "x%d" % i))
            X = torch.stack(X, 2)

        if self.n_z > 0 and self.n_y == 0 and self.n_x == 0:
            print("Black Box case: LOCALS only")
            latents = Z
        elif self.n_z == 0 and self.n_y > 0 and self.n_x == 0:
            print("Black Box case: GLOBAL CONDITIONS only")
            latents = Y
        elif self.n_z == 0 and self.n_y == 0 and self.n_x > 0:
            print("Black Box case: GLOBALS only")
            latents = X
        elif self.n_z > 0 and self.n_y > 0 and self.n_x == 0:
            print("Black Box case: LOCALS and GLOBAL CONDITIONS only")
            latents = torch.cat([Y, Z], -1)
        elif self.n_z > 0 and self.n_y == 0 and self.n_x > 0:
            print("Black Box case: LOCALS and GLOBALS only")
            latents = torch.cat([X, Z], -1)
        elif self.n_z == 0 and self.n_y > 0 and self.n_x > 0:
            print("Black Box case: GLOBALS and GLOBAL CONDITIONS only")
            latents = torch.cat([X, Y], -1)
        elif self.n_z > 0 and self.n_y > 0 and self.n_x > 0:
            #print("Black Box case: LOCALS & GLOBALS & GLOBAL CONDITIONS")
            latents = torch.cat([X, Y, Z], -1)
        else:
            raise Exception("must assign latents")
        n_latents = self.n_x + self.n_y + self.n_z

        # Neural components initialization
        n = 4 + self.n_latent_species + n_latents + self.n_treatments + self.device_depth
        states_act = decoder_state_act
        states_deg = decoder_state_deg
        #neural_precisions = NeuralPrecisions(self.nspecies, self.n_hidden_precisions,
         #                                    inputs=self.n_states + self.n_x + self.n_y + self.n_z + self.n_treatments + self.device_depth)
                                             #hidden_activation=tf.nn.relu)
        neural_precisions = decoder_neural_precisions

        def reaction_equations(state, t):
            all_reshaped_state = torch.reshape(state, [n_batch * n_iwae, self.n_states])

            # States
            reshaped_state = all_reshaped_state[:, :-4]
            ZZ_states = torch.cat([reshaped_state, \
                                   torch.reshape(latents, [n_batch * n_iwae, n_latents]), \
                                   treatments_rep, \
                                   devices], 1) # (batch_size*n_iwae,6+5+2+7)=(batch_size*n_iwae,20)
            states = states_act(ZZ_states) - states_deg(ZZ_states) * reshaped_state

            # Precisions
            reshaped_var_state = all_reshaped_state[:, -4:]
            ZZ_vrs = torch.cat([all_reshaped_state, \
                                torch.reshape(latents, [n_batch * n_iwae, n_latents]), \
                                treatments_rep, \
                                devices], 1)  # (batch_size*n_iwae,10+5+2+7)=(batch_size*n_iwae,24)  for x+h (state+hidden_state)
            vrs = neural_precisions.act(ZZ_vrs) - neural_precisions.deg(ZZ_vrs) * reshaped_var_state

            return torch.reshape(torch.cat([states, vrs], 1), [n_batch, n_iwae, self.n_states])

        return reaction_equations