
class ODEDecoder(object):
    def __init__(self, params):
        self.solver = params["solver"]
        self.ode_model = params["model"]

    def __call__(self, conds_obs, dev_1hot, times, thetas, clipped_thetas, decoder_linear_4, decoder_state_act, decoder_state_deg, decoder_neural_precisions, device, condition_on_device):
        x_sample, _f_sample = self.ode_model.simulate(
            clipped_thetas, times, conds_obs, dev_1hot, self.solver, decoder_linear_4, decoder_state_act, decoder_state_deg, decoder_neural_precisions, device, condition_on_device) #x_sample:（batch_size,n_iwae,86,10）
        # TODO: why just params here and not clipped params?
        x_predict = self.ode_model.observe(x_sample, thetas) #x_predict: (batch_size.n_iwae.86,4)
        return x_sample, x_predict