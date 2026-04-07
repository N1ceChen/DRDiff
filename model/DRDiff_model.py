import os
from copy import deepcopy
import torch
from tqdm.auto import tqdm
import time
from DRDiff.Diffusion.Res_Diffusion import ResDiffusion
from DRDiff.arch.unet import UNet, AverageMeter
from DRDiff.arch.tools import calc_mae, calc_r2, calc_rmse, calc_psnr, calc_ssim, calc_mbe
from DRDiff.arch.losses import image_compare_loss
from DRDiff.datasets import SimpleDataset, FrequencyDataset
from torch.utils.data import DataLoader

class DRDiffTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = self.configs.train['epochs']
        self.num_timesteps = self.configs.diffusion.params.get("steps")
        self.diffusion_sf = self.configs.diffusion.params.get("sf")

        self.train_dataloader = self.build_training_dataloader()
        self.val_dataloader = self.build_val_dataloader()
        self.fre_dataloader = self.build_fre_dataloader()
        self.build_model()
        self.build_diffusion_model()
        self.setup_optimization()

    def setup_optimization(self):
        self.optimizer = torch.optim.AdamW(self.unet_model.parameters(), lr=self.configs.train.get('lr'))

    def build_model(self):
        params = self.configs.model.get('params', dict)
        unet_model = UNet(**params)
        unet_model.cuda()
        if self.configs.model.ckpt_path is not None:
            state = torch.load(self.configs.model.ckpt_path, map_location=f"cuda:0")
            if 'state_dict' in state:
                state = state['state_dict']
            reload_model(unet_model, state)
        self.unet_model = unet_model

    def build_diffusion_model(self):
        diffusion_opt = self.configs.get('diffusion', dict)
        self.ResDiffusion_Model = ResDiffusion(diffusion_opt)

    def build_training_dataloader(self):
        opt = {}
        opt['max_T'] = self.configs['max_T']
        opt['sf'] = self.configs.diffusion.params['sf']
        opt['lr_path'] = self.configs.data.train.params['lr']
        opt['hr_path'] = self.configs.data.train.params['hr']
        batch_size = self.configs.train.get('batch')[0]
        num_workers = self.configs.train.get('num_workers')
        return DataLoader(SimpleDataset(opt), batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def build_val_dataloader(self):
        opt = {}
        opt['max_T'] = self.configs['max_T']
        opt['sf'] = self.configs.diffusion.params['sf']
        opt['lr_path'] = self.configs.data.val.params['lr']
        opt['hr_path'] = self.configs.data.val.params['hr']
        batch_size = self.configs.train.get('batch')[1]
        return DataLoader(SimpleDataset(opt), batch_size=batch_size, shuffle=False)

    def build_fre_dataloader(self):
        opt = {}
        opt['frequency'] = '36.5'
        opt['max_T'] = self.configs['max_T']
        opt['sf'] = self.configs.diffusion.params['sf']
        opt['lr_path'] = self.configs.data.val.params['lr']
        opt['hr_path'] = self.configs.data.val.params['hr']
        batch_size = self.configs.train.get('batch')[1]
        return DataLoader(FrequencyDataset(opt), batch_size=batch_size, shuffle=False)

    def train_step(self, epoch):
        batch = -1
        for data in self.train_dataloader:
            batch += 1
            self.optimizer.zero_grad()
            hr, lr = data['hr'].to(self.device), data['lr'].to(self.device)
            assert hr.shape==lr.shape, "Train hr and lr size error!"
            lr_ori = deepcopy(lr)
            tt = torch.randint(
                0, self.num_timesteps,
                size=(hr.shape[0],),
                device=hr.device,
            )
            noise = torch.randn(
                size=hr.shape,
                device=hr.device,
            )
            x_t = self.ResDiffusion_Model.forward_addnoise(x_start=hr, y=lr, t=tt, noise=noise)
            network_output = self.unet_model(self.ResDiffusion_Model.scale_input_resshift(inputs=x_t, t=tt),
                                             time=tt, lr=lr_ori)

            loss = image_compare_loss(hr, network_output)
            loss.backward()
            self.optimizer.step()
            if batch % self.configs.train.log_freq[0] == 0:
                print(f"Epoch {epoch}, Batch {batch}, Loss {loss.item()}")
                log_dir = f'./log'
                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
                    f.write(f"Epoch {epoch}, Batch {batch}, Loss {loss.item()}\n")

    def train(self):
        for epoch in range(self.epochs):
            epoch += 1
            print("Epoch {}/{}".format(self.epochs, epoch))
            self.train_step(epoch)
            if epoch % self.configs.train.log_freq[1] == 0:
                print(f"evaluating epoch {epoch}")
                self.evaluate()
            if epoch % self.configs.train.log_freq[2] == 0:
                torch.save(self.unet_model.state_dict(), f'./log/{epoch}.pth')

    def evaluate(self):
        max_T = 309.99
        epoch_ssim = AverageMeter()
        epoch_psnr = AverageMeter()
        epoch_rmse = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_r2 = AverageMeter()
        i = 0

        start = time.time()
        for data in self.val_dataloader:
            i += 1
            lr = data['lr'].to(self.device)
            labels = data['hr']
            lr2 = deepcopy(lr)

            indices = list(range(self.num_timesteps))[::-1]
            noise = torch.randn_like(lr)
            x_t = self.ResDiffusion_Model.prior_sample(lr, noise)
            indices = tqdm(indices)
            for t in indices:
                tt = torch.tensor([t] * x_t.shape[0], device=x_t.device)
                with torch.no_grad():
                    model_pred = self.unet_model(self.ResDiffusion_Model.scale_input_resshift(inputs=x_t, t=tt),
                                             time=tt, lr=lr2)
                    noise = torch.randn_like(model_pred)
                    x_t = self.ResDiffusion_Model.inverse_denoise(x_start=model_pred, x_t=x_t, y_0=lr, t=tt, noise=noise)
                x_t = x_t.clamp(min=-1.0, max=1.0)

            preds = x_t.cpu() * 0.5 + 0.5
            labels = labels * 0.5 + 0.5

            epoch_psnr.update(calc_psnr(preds, labels), len(lr))
            epoch_ssim.update(calc_ssim(preds, labels), len(lr))
            preds = preds * max_T
            labels = labels * max_T
            epoch_rmse.update(calc_rmse(preds, labels), len(lr))
            epoch_r2.update(calc_r2(preds, labels), len(lr))
            epoch_mae.update(calc_mae(preds, labels), len(lr))

        end = time.time()
        time_cost = (end - start) / 72.
        print("Time cost: {:.3f}".format(time_cost))
        print('eval rmse: {:.2f}, eval r2: {:.3f}, eval mae: {:.2f}, eval psnr: {:.2f}, eval ssim: {:.2f}'.format(
            epoch_rmse.avg, epoch_r2.avg, epoch_mae.avg, epoch_psnr.avg, epoch_ssim.avg))
        log_dir = f'./log'
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'eval.txt'), 'a') as f:
            f.write(f"eval rmse: {epoch_rmse.avg}, eval r2: {epoch_r2.avg}, eval mae: {epoch_mae.avg}, eval psnr: "
                    f"{epoch_psnr.avg}, eval ssim: {epoch_ssim.avg}\n")

    def evaluate_frequency(self):
        max_T = 309.99
        epoch_ssim = AverageMeter()
        epoch_psnr = AverageMeter()
        epoch_rmse = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mbe = AverageMeter()
        epoch_r2 = AverageMeter()
        i = 0
        for data in self.fre_dataloader:
            i += 1
            lr = data['lr'].to(self.device)
            labels = data['hr']
            lr2 = deepcopy(lr)

            indices = list(range(self.num_timesteps))[::-1]
            noise = torch.randn_like(lr)
            x_t = self.ResDiffusion_Model.prior_sample(lr, noise)

            indices = tqdm(indices)
            for t in indices:
                tt = torch.tensor([t] * x_t.shape[0], device=x_t.device)
                with torch.no_grad():
                    model_pred = self.unet_model(self.ResDiffusion_Model.scale_input_resshift(inputs=x_t, t=tt),
                                             time=tt, lr=lr2)
                    noise = torch.randn_like(model_pred)
                    x_t = self.ResDiffusion_Model.inverse_denoise(x_start=model_pred, x_t=x_t, y_0=lr, t=tt, noise=noise)
                x_t = x_t.clamp(min=-1.0, max=1.0)

            preds = x_t.cpu() * 0.5 + 0.5
            labels = labels * 0.5 + 0.5

            epoch_psnr.update(calc_psnr(preds, labels), len(lr))
            epoch_ssim.update(calc_ssim(preds, labels), len(lr))
            preds = preds * max_T
            labels = labels * max_T
            epoch_rmse.update(calc_rmse(preds, labels), len(lr))
            epoch_r2.update(calc_r2(preds, labels), len(lr))
            epoch_mae.update(calc_mae(preds, labels), len(lr))
            epoch_mbe.update(calc_mbe(preds, labels), len(lr))

        print('eval rmse: {:.2f}, eval r2: {:.3f}, eval mbe: {:.2f}, eval psnr: {:.2f}, eval ssim: {:.2f}'.format(
            epoch_rmse.avg, epoch_r2.avg, epoch_mbe.avg, epoch_psnr.avg, epoch_ssim.avg))

def reload_model(model, ckpt):
    module_flag = list(ckpt.keys())[0].startswith('module.')
    compile_flag = '_orig_mod' in list(ckpt.keys())[0]

    for source_key, source_value in model.state_dict().items():
        target_key = source_key
        if compile_flag and (not '_orig_mod.' in source_key):
            target_key = '_orig_mod.' + target_key
        if module_flag and (not source_key.startswith('module')):
            target_key = 'module.' + target_key

        assert target_key in ckpt
        source_value.copy_(ckpt[target_key])