
def GeneralLogImportanceWeights(log_p_observations, log_p_theta, log_q_theta, beta):
    return log_p_observations +  beta*(log_p_theta - log_q_theta)  # log [p(x | theta) p(theta) / q (theta | x)]