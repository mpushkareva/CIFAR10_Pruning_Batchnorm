import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import os 

warnings.filterwarnings('ignore')
from tqdm import tqdm_notebook

def test(network, test_loader, device, loss_function=nn.NLLLoss(),  is_train=False, is_print=True):
    if is_train:
        network.train()
    else:
        network.eval()
    test_loss = []
    correct = []
    network = network.to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss.append(loss_function(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            correct.append(pred.eq(target.data.view_as(pred)).sum().item())
    if is_print:
        print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            np.mean(test_loss), np.sum(correct), len(test_loader.dataset), 100. * np.sum(correct) / len(test_loader.dataset)))
    
    return test_loss, correct


def train_one_epoch(network, train_loader, epoch, optimizer, device, loss_function=nn.NLLLoss(), is_print=True):
    #network.train()
    #cross_ent = nn.NLLLoss() # nn.CrossEntropyLoss()
    correct = []
    train_loss = []
    correct_eval = []
    train_eval_loss = []
    log_interval = 50
    for batch_idx, (data, target) in enumerate(train_loader):
        network.train()
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct.append(pred.eq(target.data.view_as(pred)).sum().item())
        train_loss.append(loss.item())
        
        network.eval()
        output = network(data)
        loss = loss_function(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct_eval.append(pred.eq(target.data.view_as(pred)).sum().item())
        train_eval_loss.append(loss.item())
        
        
        if is_print and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    if is_print:
        print('Train set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            np.mean(train_loss), np.sum(correct), len(train_loader.dataset), 100. * np.sum(correct) / len(train_loader.dataset)))
    return train_loss, correct, train_eval_loss, correct_eval


def train_model(network, network_name, train_loader, test_loader, n_epochs, learning_rate, device):
    #optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(network.parameters(), learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    
    # сохранение результатов и обученной модели в файл
    if not os.path.isdir('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.isdir('./checkpoints/CIFAR10'):
        os.makedirs('./checkpoints/CIFAR10')
    if not os.path.isdir('./checkpoints/CIFAR10/{}'.format(network_name)):
        os.makedirs('./checkpoints/CIFAR10/{}'.format(network_name))

    train_losses_history = []
    train_eval_losses_history = []
    test_losses_history = []
    train_correct_history = []
    train_eval_correct_history = []
    test_correct_history = []
        
    test(network, test_loader, device, loss_function=nn.NLLLoss())
    for epoch in tqdm_notebook(range(n_epochs)):
        train_loss, train_correct, train_eval_loss, train_eval_correct = train_one_epoch(network, train_loader, epoch, optimizer, device, loss_function=nn.NLLLoss())
        test_loss, test_correct = test(network, test_loader, device, loss_function=nn.NLLLoss())
        lr_scheduler.step(np.mean(test_loss))
        train_losses_history.extend(train_loss)
        test_losses_history.extend(test_loss)
        train_eval_losses_history.extend(train_eval_loss)
        train_correct_history.extend(train_correct)
        train_eval_correct_history.extend(train_eval_correct)
        test_correct_history.extend(test_correct)
        # сохранение результатов и обученной модели в файл
        training_results = {
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses_history,                
                'train_eval_loss': train_eval_losses_history,
                'test_loss': test_losses_history,
                'train_correct': train_correct_history,
                'test_correct': test_correct_history,
                'train_eval_correct': train_eval_correct_history,
                'train_len': len(train_loader.dataset),
                'test_len': len(test_loader.dataset),
                'train_batchsize': train_loader.batch_size,
                'test_batchsize': test_loader.batch_size,
                }
        torch.save(training_results, f'./checkpoints/CIFAR10/{network_name}/{epoch}.pth')
      
    return training_results

def plot_loss_and_accuracy(training_results):
    fig, axs = plt.subplots(2, 2, figsize=(15,5))
    
    batchsize = training_results['train_batchsize']
    y = np.array(training_results['train_loss'])
    x = np.arange(0, batchsize * len(y), batchsize) / training_results['train_len']
    axs[0, 0].plot(x, y, color='blue', alpha=0.5)
    
    batchsize = training_results['test_batchsize']
    y = np.array(training_results['test_loss'])
    x = np.arange(0, batchsize * len(y), batchsize) / training_results['test_len']
    axs[0, 0].scatter(x, y, color='red')
    
    axs[0, 0].legend(['Train Loss', 'Test Loss'])#, loc='upper right')
    axs[0, 0].set_xlabel('epoch number')
    axs[0, 0].set_ylabel('negative log likelihood loss')
    axs[0, 0].set_title('Loss')
    
    batchsize = training_results['train_batchsize']
    y = np.array(training_results['train_eval_loss'])
    x = np.arange(0, batchsize * len(y), batchsize) / training_results['train_len']
    axs[1, 0].plot(x, y, color='blue', alpha=0.5)
    
    batchsize = training_results['test_batchsize']
    y = np.array(training_results['test_loss'])
    x = np.arange(0, batchsize * len(y), batchsize) / training_results['test_len']
    axs[1, 0].scatter(x, y, color='red')
    
    axs[1, 0].legend(['Train Eval Loss', 'Test Loss'])#, loc='upper right')
    axs[1, 0].set_xlabel('epoch number')
    axs[1, 0].set_ylabel('negative log likelihood loss')
    axs[1, 0].set_title('Loss')
    
    batchsize = training_results['train_batchsize']
    y = np.array(training_results['train_correct']) / batchsize
    x = np.arange(0, batchsize * len(y), batchsize) / training_results['train_len']
    axs[0, 1].plot(x, y, color='blue', alpha=0.5)
    
    batchsize = training_results['test_batchsize']
    y = np.array(training_results['test_correct']) / batchsize
    x = np.arange(0, batchsize * len(y), batchsize) / training_results['test_len']
    axs[0, 1].scatter(x, y, color='red')
    
    axs[0, 1].legend(['Train accuracy', 'Test accuracy'])#, loc='upper right')
    axs[0, 1].set_xlabel('epoch number')
    axs[0, 1].set_ylabel('accuracy')
    axs[0, 1].set_title('Correct')
    
    batchsize = training_results['train_batchsize']
    y = np.array(training_results['train_eval_correct']) / batchsize
    x = np.arange(0, batchsize * len(y), batchsize) / training_results['train_len']
    axs[1, 1].plot(x, y, color='blue', alpha=0.5)
    
    batchsize = training_results['test_batchsize']
    y = np.array(training_results['test_correct']) / batchsize
    x = np.arange(0, batchsize * len(y), batchsize) / training_results['test_len']
    axs[1, 1].scatter(x, y, color='red')
    
    axs[1, 1].legend(['Train eval accuracy', 'Test accuracy'])#, loc='upper right')
    axs[1, 1].set_xlabel('epoch number')
    axs[1, 1].set_ylabel('accuracy')
    axs[1, 1].set_title('Correct')
    plt.tight_layout()
    plt.show()