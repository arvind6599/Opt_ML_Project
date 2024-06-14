'''
**********************************************
Input must be a pytorch tensor
**********************************************
'''

import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
def reconstruct(net,cp_dummy_data, cp_dummy_label, gradients,tt):

    optimizer = torch.optim.LBFGS([cp_dummy_data, cp_dummy_label])
    optimizer.zero_grad()
    history = []
    for iters in range(100):
        def closure():
            optimizer.zero_grad()

            pred = net(cp_dummy_data)
            dummy_onehot_label = F.softmax(cp_dummy_label, dim=-1)
            dummy_loss = cross_entropy_for_onehot(pred, dummy_onehot_label) # TODO: fix the gt_label to dummy_label in both code and slides.
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            grad_count = 0
            for gx, gy in zip(dummy_dy_dx, gradients): # TODO: fix the variablas here
                grad_diff += ((gx - gy) ** 2).sum()
                grad_count += gx.nelement()
            # grad_diff = grad_diff / grad_count * 1000
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
        history.append(tt(cp_dummy_data[0].cpu()))

    return history

def train(model, num_epochs, testloader, trainloader, num_batches, compress_function=None, input_compress_settings=None):

  # Define loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

  # Lists to store metrics
  train_losses = []
  test_losses = []
  train_accuracies = []
  test_accuracies = []


  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      
      for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx >= num_batches:
          break

        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        # print(targets)

        outputs = model(inputs)
        # print(outputs)
        onehot_target= label_to_onehot(targets)

        # loss = criterion(outputs, targets)
        loss = cross_entropy_for_onehot(outputs,onehot_target)
        loss.backward()

        # Apply gradient compression
        if compress_function is not None and input_compress_settings is not None:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = compress_function(param.grad.data, input_compress_settings)

        optimizer.step()
        running_loss += loss.item()



      print(f'Epoch {epoch + 1}, Train_Loss: {running_loss / len(trainloader)}')
      train_losses.append(running_loss / len(trainloader))

      # Validate the model
      model.eval()
      test_loss = 0
      correct = 0
      total = 0
      with torch.no_grad():
          for inputs, targets in testloader:
              inputs, targets = inputs.cuda(), targets.cuda()
              outputs = model(inputs)
              loss = criterion(outputs, targets)
              test_loss += loss.item()
              _, predicted = torch.max(outputs, 1)
              total += targets.size(0)
              correct += (predicted == targets).sum().item()

      print(f'Epoch {epoch + 1}, Test_Loss: {test_loss / len(testloader)}')
      test_losses.append(test_loss / len(testloader))
      print(f'Accuracy: {100 * correct / total}%')
      test_accuracies.append(100 * correct / total)
      print("###########################################")

      # if (epoch%10==0):


  print('Finished Training')

  return train_losses, test_losses, test_accuracies, model



def regenerate(net, gradient, criterion, epochs,sample_data, tt):

    x_shape, y_shape = sample_data[0].size(),sample_data[1].size()

    dummy_data = torch.randn(x_shape).to(device).requires_grad_(True)
    dummy_label = torch.randn(y_shape).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label] )

    history = []
    for iters in range(epochs):
        def closure():
            optimizer.zero_grad()

            pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(pred, dummy_onehot_label) # TODO: fix the gt_label to dummy_label in both code and slides.
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            if iters == 0:
                print(dummy_dy_dx)
            grad_diff = 0

            for gx, gy in zip(dummy_dy_dx, gradient): # TODO: fix the variablas here
                grad_diff += ((gx - gy) ** 2).sum()
   
            grad_diff.backward()
            
            return grad_diff
        # print(iters)
        optimizer.step(closure)
        error_tensor = torch.abs(sample_data[0].to(device) - dummy_data.to(device))
        mean_error = torch.mean(error_tensor)
        if iters % 10 == 0:
            current_loss = closure()
            print(current_loss, mean_error)
        history.append([tt(dummy_data[0].cpu()),current_loss.item(),mean_error])  

        # plt.imshow(history[iters][0])



    return history

def deep_leakage_from_gradients(model, origin_grad): 
    '''
    This function calculates the deep leakage from gradients of a model. It takes in the model and the gradients of the model as input and returns the deep leakage.

    Parameters:
    model (torch.nn.Module): The model for which the deep leakage is to be calculated.
    origin_grad (torch.Tensor): Publicly shared gradients of the model, used to find the deep leakage.
    '''

    dummy_data = torch.randn(origin_data.size())
    dummy_label =  torch.randn(dummy_label.size())
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label] )

    for iters in range(300):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data) 
            dummy_loss = criterion(dummy_pred, F.softmax(dummy_label, dim=-1)) 
            dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = sum(((dummy_grad - origin_grad) ** 2).sum() \
                for dummy_g, origin_g in zip(dummy_grad, origin_grad))
            
            grad_diff.backward()
            return grad_diff
        
        optimizer.step(closure)
        
    return dummy_data, dummy_label


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))



def quantize(x,input_compress_settings={}):
    compress_settings={'n':6}
    compress_settings.update(input_compress_settings)
    #assume that x is a torch tensor
    
    n=compress_settings['n']
    #print('n:{}'.format(n))
    x=x.float()
    x_norm=torch.norm(x,p=float('inf'))
    
    sgn_x=((x>0).float()-0.5)*2
    
    p=torch.div(torch.abs(x),x_norm)
    renormalize_p=torch.mul(p,n)
    floor_p=torch.floor(renormalize_p)
    compare=torch.rand_like(floor_p)
    final_p=renormalize_p-floor_p
    margin=(compare < final_p).float()
    xi=(floor_p+margin)/n
    
    Tilde_x=x_norm*sgn_x*xi
    
    return Tilde_x


def uniform_quantization(x, levels):
    '''
    Perform uniform quantization on the input tensor x, with levels number of levels.

    Parameters:
    x (torch.Tensor): The input tensor to be quantized.
    levels (int): The number of levels to quantize the input tensor into.

    '''
    min_val, max_val = x.min(), x.max()
    scale = (max_val - min_val) / (levels - 1)
    quantized = torch.round((x - min_val) / scale) * scale + min_val
    return quantized

def log_quantization(tensor, base=2):
    '''
    Perform log quantization on the input tensor x, with base as the base of the logarithm.

    Parameters:
    tensor (torch.Tensor): The input tensor to be quantized.
    base (int): The base of the logarithm to be used for quantization.
    '''

    sign = torch.sign(tensor)
    log_tensor = torch.log(torch.abs(tensor) + 1e-9) / torch.log(torch.tensor(base))
    quantized = torch.round(log_tensor) * torch.log(torch.tensor(base))
    return sign * torch.exp(quantized)



def kmeans_quantization(tensor, clusters):
    '''
    Perform k-means quantization on the input tensor x, with clusters number of clusters.
    Parameters:
    tensor (torch.Tensor): The input tensor to be quantized.
    clusters (int): The number of clusters to quantize the input tensor into.
    '''
    tensor_reshaped = tensor.view(-1, 1).numpy()
    kmeans = KMeans(n_clusters=clusters).fit(tensor_reshaped)
    quantized = torch.tensor(kmeans.cluster_centers_[kmeans.labels_]).view_as(tensor)
    return quantized


def stochastic_rounding(tensor, levels):
    '''
    Stochastic rounding involves rounding to the nearest quantized value with a probability proportional to the distance from the exact value, which can preserve more information in expectation

    Parameters:
    tensor (torch.Tensor): The input tensor to be quantized.
    levels (int): The number of levels to quantize the input tensor into.
    '''

    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (levels - 1)
    scaled = (tensor - min_val) / scale
    lower = torch.floor(scaled)
    upper = torch.ceil(scaled)
    prob = scaled - lower
    quantized = torch.where(torch.rand_like(tensor) < prob, upper, lower) * scale + min_val
    return quantized

def fixed_point_quantization(tensor, num_bits, fractional_bits):
    '''
    Fixed-point quantization involves scaling the input tensor by a power of 2, rounding to the nearest integer, and then scaling back to the original range.
    '''

    scale = 2 ** fractional_bits
    quantized = torch.round(tensor * scale) / scale
    max_val = 2 ** (num_bits - fractional_bits - 1) - 1 / scale
    min_val = -max_val
    quantized = torch.clamp(quantized, min_val, max_val)
    return quantized


def add_sparsity(x, sparsity_ratio=0.2):
    """
    Adds sparsity to the input tensor by setting a specified percentage of the smallest absolute values to zero.
    
    Parameters:
    tensor (torch.Tensor): The input tensor.
    sparsity_ratio (float): The ratio of elements to be set to zero, between 0 and 1.
    
    Returns:
    torch.Tensor: The sparse tensor.
    """
    flat_tensor = x.flatten()
    k = int(sparsity_ratio * flat_tensor.size(0))

    # print("Number of elemnts that will be zeroed out",k)
    
    if k > 0:
        threshold = flat_tensor.abs().kthvalue(k).values.item()
        mask = flat_tensor.abs() > threshold
        sparse_tensor = flat_tensor * mask.float()
        return sparse_tensor.view_as(x)
    else:
        return x


