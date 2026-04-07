import math
import numpy as np
import torch

# https://github.com/zsyOAOA/ResShift
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class ResDiffusion:
    def __init__(self, configs):
        opt = configs['params']
        self.schedule_name = opt['schedule_name']
        if self.schedule_name == 'exponential':
            power_u = opt['schedule_kwargs']['power_u']
            power_eps = opt['schedule_kwargs']['power_eps']
            min_noise_level = opt['min_noise_level']
            num_diffusion_timesteps = opt['steps']
            etas_end = opt['etas_end']
            etas_start = min(min_noise_level, 0.04)
            gammas_start = min(min_noise_level, 0.04)
            increaser = math.exp(1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start))
            base = np.ones([num_diffusion_timesteps, ]) * increaser
            # u schedule
            power_u_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power_u
            power_u_timestep *= (num_diffusion_timesteps - 1)
            self.sqrt_etas = np.power(base, power_u_timestep) * etas_start
            self.etas = self.sqrt_etas ** 2
            # eps schedule
            power_eps_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power_eps
            power_eps_timestep *= (num_diffusion_timesteps - 1)
            self.sqrt_gammas = np.power(base, power_eps_timestep) * gammas_start
            self.gammas = self.sqrt_gammas ** 2
        else:
            raise KeyError(f'{self.schedule_name} is not a valid schedule')

        self.pres = opt['pres']
        self.normalize_input = opt['normalize_input']
        self.latent_flag = opt['latent_flag']
        self.num_diffusion_timesteps = num_diffusion_timesteps

        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        self.gammas_prev = np.append(0.0, self.gammas[:-1])
        self.beta = self.gammas - self.gammas_prev

        self.posterior_mean_coef1 = self.gammas_prev / self.gammas
        self.posterior_mean_coef2 = self.etas_prev - self.gammas_prev / self.gammas * self.etas
        self.posterior_mean_coef3 = self.beta / self.gammas + self.gammas_prev / self.gammas * self.etas - self.etas_prev
        self.posterior_variance = self.gammas_prev / self.gammas * self.beta

        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)

    def forward_addnoise(self, x_start, y, t, noise):
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_gammas, t, x_start.shape) * noise
        )

    def inverse_denoise(self, x_start, x_t, y_0, t, noise):
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        ) #nos noise when t==0
        output = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * y_0
            + _extract_into_tensor(self.posterior_mean_coef3, t, x_t.shape) * x_start
            + nonzero_mask * torch.exp(0.5 * _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)) * noise
        )
        return output

    def prior_sample(self, y, noise):
        t = torch.tensor([self.num_diffusion_timesteps - 1, ] * y.shape[0], device=y.device).long()
        return y + _extract_into_tensor(self.sqrt_gammas, t, y.shape) * noise

    def scale_input_resshift(self, inputs, t):
        if self.normalize_input:
            if self.latent_flag:
                std = torch.sqrt(_extract_into_tensor(self.gammas, t, inputs.shape) + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = _extract_into_tensor(self.sqrt_gammas, t, inputs.shape) * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm

from omegaconf import OmegaConf
if __name__ == "__main__":
        path = r"..\config\config.yaml"
        configs = OmegaConf.load(path)
        print(configs['diffusion']['params'])

        params = configs.model.get('params', dict)