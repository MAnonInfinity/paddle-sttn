import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from src.inpaint.sttn.auto_sttn import Discriminator
from src.inpaint.sttn.auto_sttn import InpaintGenerator
from src.tools.train.dataset_sttn import Dataset
from src.tools.train.loss_sttn import AdversarialLoss


class Trainer:
    def __init__(self, config, debug=False):
        # Trainer initialization
        self.config = config  # Store config info
        self.epoch = 0  # Current epoch
        self.iteration = 0  # Current iteration
        if debug:
            # If debug mode, set more frequent save and valid frequencies
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['iterations'] = 5

        # Set dataset and dataloader
        self.train_dataset = Dataset(config['data_loader'], split='train', debug=debug)  # Create training dataset object
        self.train_sampler = None  # Initialize train sampler to None
        self.train_args = config['trainer']  # Training parameters
        if config['distributed']:
            # If distributed training, initialize distributed sampler
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank']
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),  # Shuffle if no sampler
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler
        )

        # Set loss functions
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])  # Adversarial loss
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])  # Move loss function to device
        self.l1_loss = nn.L1Loss()  # L1 loss

        # Initialize generator and discriminator models
        self.netG = InpaintGenerator()  # Generator network
        self.netG = self.netG.to(self.config['device'])  # Move to device
        self.netD = Discriminator(
            in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge'
        )
        self.netD = self.netD.to(self.config['device'])  # Discriminator network
        # Initialize optimizers
        self.optimG = torch.optim.Adam(
            self.netG.parameters(),  # Generator parameters
            lr=config['trainer']['lr'],  # Learning rate
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2'])
        )
        self.optimD = torch.optim.Adam(
            self.netD.parameters(),  # Discriminator parameters
            lr=config['trainer']['lr'],  # Learning rate
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2'])
        )
        self.load()  # Load model

        if config['distributed']:
            # If distributed, use Distributed Data Parallel (DDP) wrapper
            self.netG = DDP(
                self.netG,
                device_ids=[self.config['local_rank']],
                output_device=self.config['local_rank'],
                broadcast_buffers=True,
                find_unused_parameters=False
            )
            self.netD = DDP(
                self.netD,
                device_ids=[self.config['local_rank']],
                output_device=self.config['local_rank'],
                broadcast_buffers=True,
                find_unused_parameters=False
            )

        # Set loggers
        self.dis_writer = None  # Discriminator writer
        self.gen_writer = None  # Generator writer
        self.summary = {}  # Store summary statistics
        if self.config['global_rank'] == 0 or (not config['distributed']):
            # If not distributed or if master node
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis')
            )
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen')
            )

    # Get current learning rate
    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

    # Adjust learning rate
    def adjust_learning_rate(self):
        # Calculate decayed learning rate
        decay = 0.1 ** (min(self.iteration, self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        # If new lr differs from current, update optimizer lr
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    # Add summary info
    def add_summary(self, writer, name, val):
        # Add and update statistics, accumulate each iteration
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        # Record every 100 iterations
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0

    # Load netG and netD models
    def load(self):
        model_path = self.config['save_dir']  # Model save path
        # Check if latest checkpoint exists
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            # Read last epoch number
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            # If no latest.ckpt, try reading model files to get latest
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()  # Sort model files to get latest
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None  # Get latest epoch value
        if latest_epoch is not None:
            # Join model file paths for netG and netD
            gen_path = os.path.join(
                model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_path = os.path.join(
                model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(
                model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            # If master node, print loading info
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_path))
            # Load generator model
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
            # Load discriminator model
            data = torch.load(dis_path, map_location=self.config['device'])
            self.netD.load_state_dict(data['netD'])
            # Load optimizer states
            data = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            # Update current epoch and iteration
            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:
            # Print warning if no model found
            if self.config['global_rank'] == 0:
                print('Warning: There is no trained model found. An initialized model will be used.')

    # Save model parameters, called each evaluation cycle
    def save(self, it):
        # Only save on rank 0 (master node)
        if self.config['global_rank'] == 0:
            # Generate file path for saving generator model state dict
            gen_path = os.path.join(
                self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
            # Generate file path for saving discriminator model state dict
            dis_path = os.path.join(
                self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
            # Generate file path for saving optimizer state dict
            opt_path = os.path.join(
                self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))

            # Print message indicating model is being saved
            print('\nsaving model to {} ...'.format(gen_path))

            # Check if the model is wrapped by DataParallel or DDP, if so get the original model
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                netD = self.netD.module
            else:
                netG = self.netG
                netD = self.netD

            # Save generator and discriminator model parameters
            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            # Save current epoch, iteration count, and optimizer state
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict()
            }, opt_path)

            # Write the latest iteration count to the "latest.ckpt" file
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))

        # Training entry point

    def train(self):
        # Initialize progress bar range
        pbar = range(int(self.train_args['iterations']))
        # If global rank 0 process, set to display progress bar
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        # Start training loop
        while True:
            self.epoch += 1  # Increment epoch count
            if self.config['distributed']:
                # If distributed training, set sampler to ensure each process gets different data
                self.train_sampler.set_epoch(self.epoch)

            # Call function to train one epoch
            self._train_epoch(pbar)
            # If iterations exceed limit in config, exit loop
            if self.iteration > self.train_args['iterations']:
                break
        # Output when training ends
        print('\nEnd training....')

        # Process input and calculate loss each cycle

    def _train_epoch(self, pbar):
        device = self.config['device']  # Get device info

        # Iterate through data loader
        for frames, masks in self.train_loader:
            # Adjust learning rate
            self.adjust_learning_rate()
            # Iteration increment
            self.iteration += 1

            # Move frames and masks to device
            frames, masks = frames.to(device), masks.to(device)
            b, t, c, h, w = frames.size()  # Get frame and mask dimensions
            masked_frame = (frames * (1 - masks).float())  # Apply mask to frames
            pred_img = self.netG(masked_frame, masks)  # Use generator to fill masked area
            # Adjust dimensions for network input
            frames = frames.view(b * t, c, h, w)
            masks = masks.view(b * t, 1, h, w)
            comp_img = frames * (1. - masks) + masks * pred_img  # Generate final composite image

            gen_loss = 0  # Initialize generator loss
            dis_loss = 0  # Initialize discriminator loss

            # Discriminator adversarial loss
            real_vid_feat = self.netD(frames)  # Discriminator identifies real frames
            fake_vid_feat = self.netD(comp_img.detach())  # Discriminator identifies fake frames (detach to avoid gradient)
            dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)  # Loss for real frames
            dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)  # Loss for fake frames
            dis_loss += (dis_real_loss + dis_fake_loss) / 2  # Average discriminator loss
            # Add discriminator loss to summary
            self.add_summary(self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())
            self.add_summary(self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
            # Optimize discriminator
            self.optimD.zero_grad()
            dis_loss.backward()
            self.optimD.step()

            # Generator adversarial loss
            gen_vid_feat = self.netD(comp_img)
            gan_loss = self.adversarial_loss(gen_vid_feat, True, False)  # Generator adversarial loss
            gan_loss = gan_loss * self.config['losses']['adversarial_weight']  # Weight amplification
            gen_loss += gan_loss  # Accumulate to generator loss
            # Add generator adversarial loss to summary
            self.add_summary(self.gen_writer, 'loss/gan_loss', gan_loss.item())

            # Generator L1 loss
            hole_loss = self.l1_loss(pred_img * masks, frames * masks)  # Calculate loss for masked area only
            # Consider mask average, multiply by hole_weight
            hole_loss = hole_loss / torch.mean(masks) * self.config['losses']['hole_weight']
            gen_loss += hole_loss  # Accumulate to generator loss
            # Add hole_loss to summary
            self.add_summary(self.gen_writer, 'loss/hole_loss', hole_loss.item())

            # Calculate L1 loss for area outside mask
            valid_loss = self.l1_loss(pred_img * (1 - masks), frames * (1 - masks))
            # Consider valid area average, multiply by valid_weight
            valid_loss = valid_loss / torch.mean(1 - masks) * self.config['losses']['valid_weight']
            gen_loss += valid_loss  # Accumulate to generator loss
            # Add valid_loss to summary
            self.add_summary(self.gen_writer, 'loss/valid_loss', valid_loss.item())

            # Generator optimization
            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # Console log output
            if self.config['global_rank'] == 0:
                pbar.update(1)  # Update progress bar
                pbar.set_description((  # Set progress bar description
                    f"d: {dis_loss.item():.3f}; g: {gan_loss.item():.3f};"  # Print loss values
                    f"hole: {hole_loss.item():.3f}; valid: {valid_loss.item():.3f}")
                )

            # Model save
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration // self.train_args['save_freq']))
            # Break condition for iteration limit
            if self.iteration > self.train_args['iterations']:
                break

