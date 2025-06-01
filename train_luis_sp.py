import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import warnings
from pathlib import Path
from torchsummary import summary
import matplotlib.pyplot as plt


from generator.DB_wav_reader import read_feats2
from generator.SR_Dataset import read_MFB_train as read_MFB

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from model.model import background_resnet
from generator.meta_generator import metaGenerator
from losses.prototypical import Prototypical


n_epochs = 80
n_shot = 3
n_query = 2
aug_percent = 0.8  # Percentage of batch to augment with SpecAugment


dataset_name = 'noisy_all_18K'  # dataset name
params_name = f'f3-s{n_shot}q{n_query}-aug{aug_percent}-1'  # parameters name for logging

root_path = Path.home().joinpath('Dropbox','DATASETS_AUDIO')                                          # recommend SSD
TRAIN_FEAT_LUIS = root_path / f'Dvectors/{dataset_name}/input_feats'
SHORT_SIZE = 100   # 100ms == 1 seconds

loss_type = 'prototypical'
use_GC = True
use_cuda = True
use_variable = True
use_checkpoint = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

seed = 100
lbl_th = 95
run_id = f'{dataset_name}_{params_name}'  # run id for logging
log_dir = 'saved_model/' + run_id
pretrained_path = 'saved_model/checkpoint_100_original.pth'  # path to pre-trained model


class SpecAugment(nn.Module):
    """
    SpecAugment implementation for PyTorch.
    
    Args:
        freq_mask_param (int): Maximum frequency mask size
        time_mask_param (int): Maximum time mask size  
        num_freq_masks (int): Number of frequency masks to apply
        num_time_masks (int): Number of time masks to apply
        p (float): Probability of applying SpecAugment to selected samples
        time_warp_param (int): Maximum time warp parameter (0 to disable)
        mask_value (float): Value to use for masking
        batch_ratio (float): Percentage of batch samples to randomly select for augmentation (0.0-1.0)
    """
    
    def __init__(self, freq_mask_param=27, time_mask_param=100, 
                 num_freq_masks=1, num_time_masks=1, p=0.8, 
                 time_warp_param=0, mask_value=0.0, batch_ratio=1.0):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p
        self.time_warp_param = time_warp_param
        self.mask_value = mask_value
        self.batch_ratio = batch_ratio  # Percentage of batch to augment
        
    def forward(self, x):
        """
        Apply SpecAugment to input spectrograms.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, freq_bins, time_steps)
                             or (batch_size, freq_bins, time_steps)
        
        Returns:
            torch.Tensor: Augmented spectrograms with same shape as input
        """
        if not self.training:
            return x
            
        if random.random() > self.p:
            return x
            
        # (batch_size, channels, freq_bins, time_steps)
        batch_size, channels, freq_bins, time_steps = x.shape         

        # Apply augmentation to each sample in the batch
        augmented = x.clone()
        
        # Randomly select which samples in the batch to augment
        num_samples_to_augment = int(batch_size * self.batch_ratio)
        if num_samples_to_augment > 0:
            # Randomly select indices
            selected_indices = random.sample(range(batch_size), num_samples_to_augment)
        else:
            selected_indices = []

        print(f"Selected {len(selected_indices)} samples for augmentation out of {batch_size} total samples.")

        for i in selected_indices:
            # # Check individual sample probability
            # if random.random() > self.p:
            #     continue
                
            # Frequency masking
            for _ in range(self.num_freq_masks):
                augmented[i, 0] = self._freq_mask(augmented[i, 0])
            
            # Time masking
            for _ in range(self.num_time_masks):
                augmented[i, 0] = self._time_mask(augmented[i, 0])
        

        return augmented
    
    def _freq_mask(self, spec):
        """Apply frequency masking to a single spectrogram."""
        freq_bins, time_steps = spec.shape
        
        if freq_bins == 0:
            return spec
        
        # Random mask size
        mask_size = random.randint(0, min(self.freq_mask_param, freq_bins))
        
        if mask_size == 0:
            return spec
        
        # Random mask position
        mask_start = random.randint(0, freq_bins - mask_size)
        mask_end = mask_start + mask_size
        
        # Apply mask
        masked_spec = spec.clone()
        masked_spec[mask_start:mask_end, :] = self.mask_value

        print(f"Applying frequency mask: start={mask_start}, end={mask_end}, size={mask_size}")
        
        return masked_spec
    
    def _time_mask(self, spec):
        """Apply time masking to a single spectrogram."""
        freq_bins, time_steps = spec.shape
        
        if time_steps == 0:
            return spec
        
        # Random mask size
        mask_size = random.randint(0, min(self.time_mask_param, time_steps))
        
        if mask_size == 0:
            return spec
        
        # Random mask position
        mask_start = random.randint(0, time_steps - mask_size)
        mask_end = mask_start + mask_size
        
        # Apply mask
        masked_spec = spec.clone()
        masked_spec[:, mask_start:mask_end] = self.mask_value

        print(f"Applying time mask: start={mask_start}, end={mask_end}, size={mask_size}")
        
        return masked_spec


# Standalone SpecAugment function for easy integration
def apply_specaugment(spectrograms, freq_mask_param=5, time_mask_param=40,
                     num_freq_masks=2, num_time_masks=2, p=0.8, 
                     time_warp_param=0, mask_value=0.0, batch_ratio=1.0):
    """
    Standalone function to apply SpecAugment to spectrograms.
    
    Args:
        spectrograms (torch.Tensor): Input spectrograms
        batch_ratio (float): Percentage of batch to augment (0.0-1.0)
        Other args: SpecAugment parameters
    
    Returns:
        torch.Tensor: Augmented spectrograms
    """
    augmenter = SpecAugment(
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        num_time_masks=num_time_masks,
        p=p,
        time_warp_param=time_warp_param,
        mask_value=mask_value,
        batch_ratio=batch_ratio
    )
    
    # Set to training mode to enable augmentation
    augmenter.train()
    
    with torch.no_grad():
        return augmenter(spectrograms)


def plot_loss_accuracy(train_losses, train_losses_e, train_losses_g, train_acc_e, train_acc_g, log_dir, run_id):
    # Plot episode loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_e, label='Episode Loss (loss_e)')
    plt.plot(train_losses_g, label='Global Loss (loss_g)')
    plt.plot(train_losses, label='Regular Loss (loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_e, label='Episode Accuracy (acc_e)')
    plt.plot(train_acc_g, label='Global Accuracy (acc_g)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{run_id}_plot.png'))
    plt.close()

def train(train_generator, model, objective, optimizer, n_episode, log_dir, scheduler, aug_percent=0.8):

    # switch to train mode
    model.train()

    patience=50
    early_stop = False
    best_loss = float('inf')
    epochs_no_improve = 0


    # for batch_idx, (data) in enumerate(train_loader):
    log_interval = int(n_episode / 2)
    avg_train_losses = []
    avg_train_losses_e = []
    avg_train_losses_g = []
    avg_train_acc_e = []
    avg_train_acc_g = []
    for t, (data) in train_generator:
        epoch = int(t / n_episode)


        if t % n_episode == 0:
            losses = AverageMeter()
            losses_e = AverageMeter()
            losses_g = AverageMeter()
            accuracy_e = AverageMeter()
            accuracy_g = AverageMeter()

        inputs, targets_g = data  # target size:(batch size), input size:(batch size, 1, n_filter, T)

        if aug_percent > 0:
            inputs = apply_specaugment(inputs, batch_ratio=aug_percent)

        targets_e = tuple([i for i in range(nb_class_train)]) * (n_query)
        targets_e = torch.tensor(targets_e, dtype=torch.long).cuda(non_blocking=True)
        support, query = split_support_query(inputs)

        loss, loss_e, loss_g, acc_e, acc_g =\
            objective(support, query, targets_g, targets_e, model, use_GC)
        losses.update(loss.item(), query.size(0))
        losses_e.update(loss_e.item(), query.size(0))
        losses_g.update(loss_g.item(), inputs.size(0))
        accuracy_e.update(acc_e * 100, query.size(0))
        accuracy_g.update(acc_g * 100, inputs.size(0))

        # episode number in epoch
        ith_episode = t % n_episode

        # Print t and ith_episode
        # print(f'Epoch: {epoch}, Episode: {t % n_episode}/{n_episode}, Ith Episode: {ith_episode}')

        if ith_episode % log_interval == 0:
            print(
                'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                'Loss {loss.avg:.4f} (loss_e: {loss_e.avg:.4f} / loss_g: {loss_g.avg:.4f})\t'
                'Acc e / g {acc_episode.avg:.4f} / {acc_global.avg:.4f}'.format(
                epoch, ith_episode, n_episode, 100. * ith_episode / n_episode,
                loss=losses, loss_e=losses_e, loss_g=losses_g, acc_episode=accuracy_e, acc_global=accuracy_g))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % n_episode == 0 and t != 0: #epoch interval
            scheduler.step(losses.avg, epoch)

            # calculate average loss and accuracy over an epoch
            avg_train_losses.append(losses.avg)
            avg_train_losses_e.append(losses_e.avg)
            avg_train_losses_g.append(losses_g.avg)
            avg_train_acc_e.append(accuracy_e.avg)
            avg_train_acc_g.append(accuracy_g.avg)

            # Early stopping check
            if losses.avg < best_loss:
                best_loss = losses.avg
                epochs_no_improve = 0
                # Save best model
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           '{}/bestCP_{}.pth'.format(log_dir, str(epoch).zfill(3)))
            else:
                epochs_no_improve += 1
                print(f'No improvement in loss for {epochs_no_improve} epoch(s).')
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}')
                    early_stop = True
                    break

    # Save last checkpoint
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                '{}/lastCP_{}.pth'.format(log_dir, str(epoch).zfill(3)))

    # find position of lowest training loss
    minposs = avg_train_losses.index(min(avg_train_losses)) + 1
    print('Lowest training loss at epoch %d' % minposs)

    # Plot and save loss/accuracy
    plot_loss_accuracy(avg_train_losses, avg_train_losses_e, avg_train_losses_g, avg_train_acc_e, avg_train_acc_g, log_dir, run_id)

    return early_stop

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_model(n_classes):
    print('=> loading checkpoint')

    # Initialize the model
    model = background_resnet(num_classes=n_classes, backbone='resnet34')

    # Load the checkpoint
    checkpoint = torch.load(pretrained_path)

    # Load pre-trained weights, ignoring mismatched layers
    model.load_state_dict(checkpoint, strict=False)

    # Freeze specified layers
    for name, module in [
        ('conv1', model.pretrained.conv1),
        ('bn1', model.pretrained.bn1),
        ('relu', model.pretrained.relu),
        ('layer1', model.pretrained.layer1),
        ('layer2', model.pretrained.layer2),
        ('layer3', model.pretrained.layer3),
    ]:
        for param in module.parameters():
            param.requires_grad = False

    # # Freeze all parameters
    # for param in model.parameters():
    #     param.requires_grad = False

    # Reinitialize the final layer to match the new number of classes
    model.weight = Parameter(torch.Tensor(n_classes, 256))
    nn.init.xavier_uniform_(model.weight)
    model.weight.requires_grad = True  # Only train the last layer


    # Initialize the optimizer
    optimizer = create_optimizer(model)

    # Move optimizer tensors to GPU if necessary
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    return model, optimizer



## lr from checkpoint = 1.0e-5
def create_optimizer(model, new_lr=1e-1, wd=1e-4):
    # setup optimizer
    optimizer = optim.SGD([
        {'params': model.parameters(), 'weight_decay': wd}
    ], lr=new_lr, momentum=0.9, nesterov=True, dampening=0)

    # # RMSprop
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99, eps=1e-8)

    return optimizer


def split_support_query(inputs):
    B, C, Fr, T = inputs.size()
    inputs = inputs.reshape(n_shot + n_query, nb_class_train, C, Fr, T)
    support = inputs[:n_shot].reshape(-1, C, Fr, T)
    query = inputs[n_shot:].reshape(-1, C, Fr, T)

    if use_variable:
        min_win, max_win = SHORT_SIZE, T
        win_size = random.randrange(min_win, max_win)
        query = query[:, :, :, :win_size].contiguous()

    return support, query

if __name__ == '__main__':


    # Load dataset
    all_train_DB, n_data, n_classes = read_feats2(TRAIN_FEAT_LUIS, n_shot, n_query)

    # Print dataset information
    print('Training set size: %d' % n_data)
    print('Number of classes: %d' % n_classes)

    # # Print number of samples available in each class
    # label_counts = all_train_DB['labels'].value_counts()

    # print(label_counts)

    # # Get labels that occur more than lbl_th times
    # valid_labels = label_counts[label_counts > lbl_th].index

    # # Filter the original DataFrame
    # train_DB = all_train_DB[all_train_DB['labels'].isin(valid_labels)]

    # # Count the number of different labels in train_DB
    # filtered_label_counts = train_DB['labels'].value_counts()
    # nb_class_train = len(filtered_label_counts)
    # print('Number of training samples per class: %d' % nb_class_train)

    #### By-pass filtering for now
    nb_class_train = n_classes  # Use all classes for training
    train_DB = all_train_DB  # Use the entire dataset for training

    nb_samples_per_class = n_shot + n_query

    n_episode = int(n_data / ((nb_samples_per_class) * nb_class_train))

    print('Number of episodes per epoch: %d' % n_episode)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate model and optimizer
    if use_checkpoint:
      model, optimizer = load_model(n_classes)
    else:
      model = background_resnet(num_classes=n_classes)
      optimizer = create_optimizer(model)

    # define objective function, optimizer and scheduler
    objective = Prototypical()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, min_lr=1e-5, threshold=1e-4, verbose=1)

    model.cuda()
    # summary(model, input_size=(1, 40, 256))

    train_generator = metaGenerator(train_DB, read_MFB,
                                    nb_classes=nb_class_train, nb_samples_per_class=nb_samples_per_class,
                                    max_iter=n_episode * n_epochs, xp=np)
    # training
    early_stopped = train(train_generator, model, objective, optimizer, n_episode, log_dir, scheduler, aug_percent=aug_percent)

    if early_stopped:
        print("Training stopped early due to no improvement in loss (early stopping).")