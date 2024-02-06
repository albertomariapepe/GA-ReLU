import torch


in_channels = 1
shape = 128

def OneStepLoss(output, target):
    loss = torch.nn.functional.mse_loss(output, target, reduction='mean')
    return loss


def ScalarLoss(output, target):
    if output.dim() == 5:
        loss = torch.nn.functional.mse_loss(output[:,:,:,:,0], target[:,:,:,:,0], reduction='mean')
    else:
        loss = torch.nn.functional.mse_loss(output[:,0,:,:], target[:,0,:,:], reduction='mean')

    return loss

def VectorLoss(output, target):

    if output.dim() == 5:
        loss = torch.nn.functional.mse_loss(output[:,:,:,:,1:], target[:,:,:,:,1:], reduction='mean')
    else:
        loss = torch.nn.functional.mse_loss(output[:,1:,:,:], target[:,1:,:,:], reduction='mean')
    return loss


def RollOutLoss(output, target, X, model, rollout = 5):

    loss = 0

    if output.dim() == 5:


        loss += torch.nn.functional.mse_loss(output, target[:,0,:,:,:],  reduction='mean')
        output_next = model(torch.stack([X[:,k,:,:,:], output]))

    for k in range(1,rollout):
        
        loss += torch.nn.functional.mse_loss(output_next, target[:,k,:,:,:], reduction='mean')
        X = output
        output = output_next
        model(torch.stack([X, output]))

    return loss/rollout

