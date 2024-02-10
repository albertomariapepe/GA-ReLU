import torch.nn.functional as F
import torch
import sys
sys.path.append(r"/home/***/Desktop/navierstokes/")

from cliffordlayers.cliffordlayers.cliffordalgebra import CliffordAlgebra

ga2d = CliffordAlgebra([1, 1])

def CliffordGELU2D(x):
    
    x0 = ga2d.embed(x, [0, 1, 2])
    
    x1 = ga2d.geometric_product(x0, ga2d.reverse(x0))
 
    x1 = ga2d.get(x1, blade_index=[0])
   
    x2 = F.relu(x1)
   
    return x2*x

def GradewiseCliffordGELU2D(x):
    x0 = ga2d.embed(x, [0, 1, 2])

    
    
    x_s = x0
    x_s[:,:,:,:,1] = 0
    x_s[:,:,:,:,2] = 0
    x_s1 = ga2d.geometric_product(x_s, ga2d.reverse(x_s))
    x_s1 = ga2d.get(x_s1, blade_index=[0])
    x_s = F.relu(x_s1) * x_s


    x_v = x0
    x_v[:,:,:,:,0] = 0
    x_v1 = ga2d.geometric_product(x_v, ga2d.reverse(x_v))
    x_v1 = ga2d.get(x_v1, blade_index=[0])
    x_v = F.relu(x_v1) * x_v

    x = ga2d.get((x_s+x_v), blade_index = [0, 1, 2])


    '''
    x_b0 = ga2d.get(x, blade_index = [3])
    x_b = ga2d.embed(x_b0, [3])
    x_b = ga2d.geometric_product(x_b, ga2d.reverse(x_b))
    x_b = ga2d.get(x_b, blade_index=[0])
    x_b = F.gelu(x_b) * x_b0
    '''
    return x


x = torch.rand((18, 3, 10, 10, 10, 3))
y = GradewiseCliffordGELU2D(x)


def GArelu(x):

    x_e1 = x[:,:,:,:,1]
    x_e2 = x[:,:,:,:,2]
    
    K = 0.5*(1 + torch.cos(torch.arctan2(x_e2, x_e1)))
    
    x[:,:,:,:,1] = K*x_e1
    x[:,:,:,:,2] = K*x_e2

    return F.relu(x)
