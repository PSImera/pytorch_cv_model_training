import os
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from tqdm import tqdm

from src.utils import transform

class FC2LNN(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer_1 = nn.Linear(input, 256, bias=False)
        self.batchnorm = nn.BatchNorm1d(256)
        self.act = nn.ReLU()
        self.layer_2 = nn.Linear (256, output)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        return x

class MyNN(nn.Module):
    def __init__(self, in_channels, output, activator='relu', input_size=(64, 64)):
        super(MyNN, self).__init__()
        self.activators = nn.ModuleDict({
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()   
        })
        self.act = self.activators[activator]
        self.layers = nn.ModuleList()

        out_to_fc = 64
        output_size = self.calc_conv_out_size(input_size)
        output_size = self.calk_pool_out_size(output_size, kernel=2)
        output_size = self.calc_conv_out_size(output_size)
        h, w = self.calk_pool_out_size(output_size, kernel=2)
        fc_in = out_to_fc * h * w

        self.layers.add_module('conv_1', nn.Conv2d(in_channels, 32, kernel_size=3))
        self.layers.add_module('bn_1', nn.BatchNorm2d(32))
        self.layers.add_module('act_1', self.act)
        self.layers.add_module('pool_1', nn.MaxPool2d(2))

        self.layers.add_module('conv_2', nn.Conv2d(32, out_to_fc, kernel_size=3))
        self.layers.add_module('bn_2', nn.BatchNorm2d(out_to_fc))
        self.layers.add_module('act_2', self.act)
        self.layers.add_module('pool_2', nn.MaxPool2d(2))

        self.layers.add_module('flatten', nn.Flatten())
        self.layers.add_module('fc1', nn.Linear(fc_in, 128))
        self.layers.add_module('act_3', self.act)
        self.layers.add_module('dropout', nn.Dropout(0.3))
        self.layers.add_module('output', nn.Linear(128, output))

    def calc_conv_out_size(self, input_size, kernel=3, stride=1, padding=0, dilation=1):
        h, w = input_size
        kw = kh = kernel
        sw = sh = stride
        ph = pw = padding
        dw = dh = dilation
        h_out = (h + 2*ph - dh*(kh-1) - 1) // sh + 1
        w_out = (w + 2*pw - dw*(kw-1) - 1) // sw + 1
        return h_out, w_out
    
    def calk_pool_out_size(self, input_size, kernel=3, stride=None, padding=0, dilation=1, ceil_mode=False):
        h, w = input_size
        if stride is None:
            stride = kernel
        kw = kh = kernel
        sw = sh = stride
        ph = pw = padding
        dw = dh = dilation
        if ceil_mode:
            h_out = math.ceil((h + 2*ph - dh*(kh-1) - 1) / sh) + 1
            w_out = math.ceil((w + 2*pw - dw*(kw-1) - 1) / sw) + 1
        else:
            h_out = (h + 2*ph - dh*(kh-1) - 1) // sh + 1
            w_out = (w + 2*pw - dw*(kw-1) - 1) // sw + 1
        
        return h_out, w_out

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MyModel():
    def __init__(self, 
                 dataset,
                 model, 
                 optimizer, 
                 loss_function,
                 sheduler,
                 save_path,
                 name,
                 classification=True,
                 is_reshape=False,
                 earlystoper=None,
                 split_list=[0.7, 0.15, 0.15],
                 batch=16,
                 save_trashold=0.0001,
                 plot=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 offload=False,
                 ):
        
        self.batch = batch
        self.shape = dataset.shape
        
        self.classification = classification
        if self.classification:
            self.classes_to_idx = dataset.classes_to_idx
            self.idx_to_classes = dataset.idx_to_classes
        self.is_reshape = is_reshape

        self.split_dataset(dataset, split_list)

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.sheduler = sheduler
        self.earlystoper = earlystoper
        self.save_path = save_path
        self.name = name
        self.epochs = 50
        self.device = device
        self.offload = offload
        self.plot = plot
        self.save_trashold = save_trashold
        
        self.best_loss = None
        if self.plot:
            self.train_loss_hist = []
            self.train_acc_hist = []
            self.val_loss_hist = []
            self.val_acc_hist = []
            self.lr_hist = []

    def split_dataset(self, dataset, split_list):
        train, val, test = random_split(dataset, split_list)
        self.train_loader = DataLoader(train, batch_size=self.batch, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=self.batch, shuffle=False)
        self.test_loader = DataLoader(test, batch_size=self.batch, shuffle=False)

    def train(self, loader=None):
        if loader is None:
            loader = self.train_loader
        self.model.train()
        acc, mean_loss = self.cycle(loader, 'train')

        if self.plot:
            self.train_loss_hist.append(mean_loss)
            self.train_acc_hist.append(acc)
        
        return acc, mean_loss
        
    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        with torch.no_grad():
            acc, mean_loss = self.cycle(loader, 'val')

        if self.plot:
            self.val_loss_hist.append(mean_loss)
            self.val_acc_hist.append(acc)

        return acc, mean_loss
    
    def test(self, loader=None):
        if loader is None:
            loader = self.test_loader
        self.model.eval()
        with torch.no_grad():
            acc, loss = self.cycle(loader, 'test')
            print(f'Mean Loss: {loss:.4f}')
            print(f'Accuracy: {acc:.4f}')
            return acc, loss
    
    def cycle(self, loader, name):
        running_loss = []
        correct_predictions = 0
        total_samples = 0

        tqdm_loop = tqdm(loader, leave=False)
        for x, y in tqdm_loop:
            if self.is_reshape:
                x = x.reshape(-1, self.shape[0] * self.shape[1])
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            loss = self.loss_function(pred, y)

            if name == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)

            if self.classification:
                correct_predictions += (pred.argmax(dim=1) == y).sum().item()
            else:
                correct_predictions += (torch.round(pred) == y).all(dim=1).sum().item()
            total_samples += y.size(0)

            accuracy = correct_predictions / total_samples if total_samples > 0 else 0

            if name == 'test':
                tqdm_loop.set_description(f'{name}_loss: {mean_loss:.4f} | {name}_acc: {accuracy:.4f}')
            else:
                tqdm_loop.set_description(f'Epoch {self.step}/{self.epochs} | {name}_loss: {mean_loss:.4f} | {name}_acc: {accuracy:.4f}')
        
        return accuracy, mean_loss

    def graph_compare(self, i,  name, *arg):
        self.axs[i].plot(arg[0])
        if len(arg) == 2:
            self.axs[i].plot(arg[1])
            self.axs[i].legend([f'{name}_train', f'{name}_val'])
        else:
            self.axs[i].legend(name)
        self.axs[i].set_title(name)
        self.axs[i].set_xlabel('Epochs')
        self.axs[i].set_xlim(0, self.epochs)
        self.axs[i].xaxis.set_major_locator(MultipleLocator(5))
        self.axs[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        self.axs[i].yaxis.set_minor_locator(AutoMinorLocator(5))
        self.axs[i].grid(which='major', color='#CCCCCC', linestyle='--', alpha=0.4)
        self.axs[i].grid(which='minor', color='#CCCCCC', linestyle=':', alpha=0.3)
    
    def plot_show(self):
        fig, self.axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
        self.graph_compare(0, 'Loss', self.train_loss_hist, self.val_loss_hist)
        self.graph_compare(1, 'Accuracy', self.train_acc_hist, self.val_acc_hist)
        self.graph_compare(2, 'Learning Rate', self.lr_hist)
        plt.tight_layout()
        plt.show()
    
    def __call__(self, epochs=50):
        self.epochs = epochs
        self.model.to(self.device)
        for epoch in range(self.epochs):
            self.step = epoch + 1
            train_acc, mean_train_loss = self.train()
            val_acc, mean_val_loss = self.validate()
            
            self.sheduler.step(mean_val_loss)
            lr = self.sheduler.get_last_lr()[0]
            self.lr_hist.append(lr)

            print(f'Epoch {self.step}/{self.epochs}, train [loss: {mean_train_loss:.4f} | acc: {train_acc:.4f}], val [loss: {mean_val_loss:.4f} | acc: {val_acc:.4f}], lr={lr:.4f}')

            if self.best_loss is None:
                self.best_loss = mean_val_loss
            if mean_val_loss < self.best_loss - self.best_loss*self.save_trashold:
                self.best_loss = mean_val_loss
                self.save_model(epoch=self.step)

            if self.earlystoper and self.earlystoper(mean_val_loss):
                print(f"The model has reached a performance plateau and is no longer learning. Stopping training")
                break

        if self.offload:
            self.model.to('cpu')
        if self.plot:
            self.plot_show()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_model'])
        self.model.to(self.device)
        print(f'Model weights loaded from {model_path}')

    def save_model(self, save_path=None, name=None, epoch=0):
        checkpoint = {
                    'state_model': self.model.state_dict(),
                    'state_optimizer': self.optimizer.state_dict(),
                    'state_scheduler': self.sheduler.state_dict(),
                    'loss': {
                        'train_loss': self.train_loss_hist,
                        'val_loss': self.val_loss_hist,
                        'best loss': self.best_loss
                    },
                    'metric': {
                        'train_acc': self.train_acc_hist,
                        'val_acc': self.val_acc_hist,
                        'Ir': self.lr_hist
                    },
                    'save_epoch': self.step
                }
        if self.classification:
            checkpoint['classes_to_idx'] = self.classes_to_idx
            checkpoint['idx_to_classes'] = self.idx_to_classes

        if save_path is None:
            save_path = self.save_path
        if name is None:
            name = self.name
        if not epoch:
            name_format = f'{name}_ckpt.pt'
        else:
            name_format = f'{name}_ckpt_epoch_{epoch}.pt'

        model_path = os.path.join(save_path, name_format)
        
        torch.save(checkpoint, model_path)
        print(f'Model seved on {epoch} as {model_path}')
    
    def inference(self, img):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            x = transform(img, norm=False).unsqueeze(0)
            if self.is_reshape:
                x = x.reshape(-1, self.shape[0] * self.shape[1])
            x = x.to(self.device)
            answer = self.model(x)
            if self.offload:
                self.model.to('cpu')

            if self.classification:
                return self.idx_to_classes[torch.argmax(answer, dim=1)[0].item()]

            return [round(c.item()) for c in answer[0]]
        