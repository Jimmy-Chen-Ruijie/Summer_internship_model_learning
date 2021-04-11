import torch

def modified_euler_integrate(d_states_d_t, init_state, times):

    x = [init_state]
    h = times[1] - times[0]
    F = []
    for t2, t1 in zip(times[1:], times[:-1]):
        f1 = d_states_d_t(x[-1], t1)
        f2 = d_states_d_t(x[-1] + h * f1, t2)

        # TODO:
        x.append(x[-1] + 0.5 * h * (f1 + f2))
        F.append(0.5 * h * (f1 + f2))

    return torch.stack(x, -1), torch.stack(F, -1)
