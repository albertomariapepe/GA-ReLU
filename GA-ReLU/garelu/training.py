from networks import (
        CliffordResNet, CliffordResNetRot, 
        CliffordResNetNorm, CliffordFourierNet, 
        CliffordFourierNetNorm, CliffordFourierNetRot,
        ResNet, FourierNet
)

from lossfunctions import (
    ScalarLoss,VectorLoss, OneStepLoss,RollOutLoss
    )

from torchsummary import summary
import torch
import optparse
import numpy as np
import logging
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import time
import os

#source ~/anaconda3/etc/profile.d/conda.sh to activate environmene
#conda activate cliffordlayers


inchan = 2
outchan = 1

parser = optparse.OptionParser()

if __name__ == '__main__':
    parser.add_option('-m', '--model',
            action="store", dest="model",
            help="training model, can choose between: R, CR, CRN, CRR, F, CF, CFN, CFR", default="R")

    parser.add_option('-A', '--activation',
            action="store", dest="activation",
            help="1 for Clifford activation functions (ours)", default= 0)
    
    parser.add_option('-t', '--trajectories',
            action="store", dest="trajectories",
            help="number of training trajectories ", default= 15600)
    
    parser.add_option('-e', '--epochs',
            action="store", dest="epochs",
            help="number of training epochs ", default= 200)
    
    parser.add_option('-b', '--batchsize',
            action="store", dest="batchsize",
            help="number of batches ", default= 16)

    parser.add_option('-l', '--learnrate',
            action="store", dest="lr",
            help="learningrate ", default= 1e-4)
    
    parser.add_option('-p', '--patience',
            action="store", dest="patience",
            help="patience for early stopping", default= 15)
    
    parser.add_option('-s', '--seed',
            action="store", dest="seed",
            help="random seed choice", default= 333)
    
    parser.add_option('-D', '--traindataset',
            action="store", dest="traindataset",
            help="train dataset folder", default= "datasets/trainingdataviscous/")
    
    parser.add_option('-V', '--valataset',
            action="store", dest="valdataset",
            help="validation dataset folder", default= "datasets/validationdataviscous/")
    
    parser.add_option('-T', '--testdataset',
            action="store", dest="testdataset",
            help="test dataset folder", default= "datasets/testdataviscous/")
    

    options, args = parser.parse_args()

    options.trajectories = int(options.trajectories)

    logfilename = "results/navierstokes_" + str(options.activation) +'CliffAct_'+ options.model + "_" +  str(options.batchsize) + "_" + str(options.trajectories) + "_" + str(options.seed) + "_" + str(options.lr) + ".log"

    if os.path.exists(logfilename):
        os.remove(logfilename)

    logging.basicConfig(filename=logfilename, level=logging.INFO)

    logging.info('*************')
    logging.info('             ')
    logging.info(f'model chosen: {options.model}')
    logging.info(f'multivector activation function?: {options.activation}')
    logging.info(f'training trajectories: {options.trajectories}')
    logging.info(f'training epochs: {options.epochs}')
    logging.info(f'training batchsize: {options.batchsize}')
    logging.info(f'learning rate: {options.lr}')
    logging.info(f'seed: {options.seed}')
    logging.info(f'patience: {options.patience}')
    logging.info(f'training dataset path: {options.traindataset}')
    logging.info(f'test dataset path: {options.testdataset}')
    logging.info('             ')
    logging.info('*************')

    import gc

    gc.collect()
    torch.cuda.empty_cache() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    


    #print(logfilename)
    

    if options.model == 'R':
        model = ResNet(inchan, outchan)
    elif options.model == 'CR':
        model = CliffordResNet(inchan, outchan, options.activation)
    elif options.model == 'CRR':
        model = CliffordResNetRot(inchan, outchan)
    elif options.model == 'CRN':
        model = CliffordResNetNorm(inchan, outchan, options.activation)
    elif options.model == 'F':
        model = FourierNet(inchan, outchan)
    elif options.model == 'CF':
        model = CliffordFourierNet(inchan, outchan, options.activation)
    elif options.model == 'CFN':
        model = CliffordFourierNetNorm(inchan, outchan)
    elif options.model == 'CFR':
        model = CliffordFourierNetRot(inchan, outchan)
    else:
        logging.error("The input model is invalid.")
  

    torch.manual_seed(options.seed)
    
    model = model.to(device)

    loss_fn = OneStepLoss

    strike = 0
   

    #Train Data
    X = np.load(options.traindataset + "X_trajectories.npy")
    Y = np.load(options.traindataset + "Y_trajectories.npy")

    
    logging.info(X[0:2,0,0:4,0:4])

    
    logging.info('****')
   

    if options.model == 'R' or options.model == 'F':
        X = np.reshape(X, (-1, inchan*3, 128, 128))
        Y = np.reshape(Y, (-1, outchan*3, 128, 128))
        logging.info(X[0,0,:4,0:4])
    else:
        X = np.transpose(X, (0,2,3,1))
        Y = np.transpose(Y, (0,2,3,1))
        X = np.reshape(X, (-1, inchan, 128, 128, 3), order = 'C')
        Y = np.reshape(Y, (-1, outchan, 128, 128, 3), order = 'C')
        logging.info(X[0,:,0:4,0:4,0])

  
    #setting the size of the training set
    X = X[:int(options.trajectories)]
    Y = Y[:int(options.trajectories)]


    numbatches = len(X) // int(options.batchsize)
   

    tensorX = torch.Tensor(X)
    tensorY = torch.Tensor(Y)

    logging.info(f"Train input shape: {tensorX.shape}")
    logging.info(f"Train labels shape: {tensorY.shape}")
    logging.info("                   ")

    dataset = TensorDataset(tensorX,tensorY) # create your datset
    traindataloader = DataLoader(dataset, batch_size=int(options.batchsize), shuffle=True)


    #Val Data
    vX = np.load(options.valdataset + "X_trajectories.npy")
    vY = np.load(options.valdataset + "Y_trajectories.npy")

    

    if options.model == 'R' or options.model == 'F':
        vX = np.reshape(vX, (-1,inchan*3, 128, 128))
        vY = np.reshape(vY, (-1, outchan*3, 128, 128))
    else:
        vX = np.transpose(vX, (0,2,3,1))
        vY = np.transpose(vY, (0,2,3,1))
        vX = np.reshape(vX, (-1, inchan, 128, 128, 3), order = 'C')
        vY = np.reshape(vY, (-1, outchan, 128, 128, 3), order = 'C')

    tensorvX = torch.Tensor(vX)
    tensorvY = torch.Tensor(vY)

    logging.info(f"Val input shape: {tensorvX.shape}")
    logging.info(f"Val labels shape: {tensorvY.shape}")

    vdataset = TensorDataset(tensorvX,tensorvY) # create your datset
    valdataloader = DataLoader(vdataset, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(options.lr))

    logging.info('*************')
    logging.info('             ')
    logging.info("Starting to train....")

    timestamp = datetime.now().strftime('%Y%m%d')
    
    best_vloss = 10e9

    model.train(True)

    for epoch in range(options.epochs):
        
        running_loss = 0 
        avg_loss = 0
        running_vloss = 0
        start = time.time()

        for i, data in enumerate(traindataloader):

            #logging.info('batch {}'.format(i + 1))
            x, y = data

            x = x.to(device)
            y = y.to(device)
            
            
            pred = model(x)
            loss = loss_fn(pred, y)

            loss.backward()
  
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()


            if i % 100  == 99:
                avg_loss = running_loss / 100 # loss per batch
                logging.info('  batch {} loss: {}'.format(i + 1, avg_loss))
                tb_x = epoch * len(traindataloader) + i + 1
                logging.info(f"LOSS train' {avg_loss}, {tb_x}")
                running_loss = 0
            

        with torch.no_grad():
            for j, vdata in enumerate(valdataloader):
                vx, vy = vdata

                vx = vx.to(device)
                vy = vy.to(device)

                vpred = model(vx)
                vloss = loss_fn(vpred, vy)
                running_vloss += vloss

        
            avg_vloss = running_vloss / (j + 1)

        end = time.time()
        logging.info(f"Epoch: {epoch} - LOSS train: {avg_loss} LOSS val: {avg_vloss} - Elapsed time: {end-start} s")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'trainedmodels/model_{}_{}_{}_{}_{}_{}'.format(options.model, options.activation, options.batchsize, options.seed, options.trajectories, timestamp)
            torch.save(model.state_dict(), model_path)
        
        else:
            strike += 1
        
        if strike == options.patience:
            break

 

    #model_path = 'trainedmodels/model_CR__32_20231017_48'
    #loading the model with lowest validation loss
    model.load_state_dict(torch.load(model_path))

    
    tX = np.load(options.testdataset + "X_trajectories.npy")
    tY = np.load(options.testdataset + "Y_trajectories.npy")



    if options.model == 'R' or options.model == 'F':
        tX = np.reshape(tX, (-1, inchan*3, 128, 128))
        tY = np.reshape(tY, (-1, outchan*3, 128, 128))
    else:
        tX = np.transpose(tX, (0,2,3,1))
        tY = np.transpose(tY, (0,2,3,1))
        tX = np.reshape(tX, (-1, inchan, 128, 128, 3), order = 'C')
        tY = np.reshape(tY, (-1, outchan, 128, 128, 3), order = 'C')



    testX = torch.Tensor(tX)
    testY = torch.Tensor(tY)

    dataset = TensorDataset(testX,testY) # create your datset
    testdataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    logging.info('*************')
    logging.info('             ')
    logging.info("Starting to test....")
    logging.info('             ')

    model.eval() 

    Os = 0
    Sc = 0
    Ve = 0

    with torch.no_grad():
        for i, data in enumerate(testdataloader):
            x, y = data

            x = x.to(device)
            y = y.to(device)

            predY = model(x)

            Os += OneStepLoss(predY, y)
            Sc += ScalarLoss(predY, y)
            Ve += VectorLoss(predY, y)
            #Ro += RollOutLoss(predY, y, x, model)

    logging.info(f"OneStep loss: {Os/i}")
    logging.info(f"Scalar loss: {Sc/i}")
    logging.info(f"Vector loss: {Ve/i}")
    #logging.info(f"Rollout loss: {RollOutLoss(predY, testY, testX, model)}")


    logging.info('             ')
    logging.info("Done.")


    

