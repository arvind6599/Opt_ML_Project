'''
**********************************************
Input must be a pytorch tensor
**********************************************
'''

import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F


def deep_leakage_from_gradients(model, origin_grad): 

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
    
    return  dummy_data, dummy_label


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
    sign = torch.sign(tensor)
    log_tensor = torch.log(torch.abs(tensor) + 1e-9) / torch.log(torch.tensor(base))
    quantized = torch.round(log_tensor) * torch.log(torch.tensor(base))
    return sign * torch.exp(quantized)



def kmeans_quantization(tensor, clusters):
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


def deep_leakage_from_gradients(model, origin_grad): 

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