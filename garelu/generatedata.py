from phi.flow import *
import logging
import os

PATH  = 'datasets/trainingdataviscous/'
logfilename = 'generatedata.log'

if os.path.exists(logfilename):
    os.remove(logfilename)

logging.basicConfig(filename=logfilename, level=logging.INFO)

backend.default_backend().list_devices('GPU')
assert backend.default_backend().set_default_device('GPU')


xsize = ysize = 128
resolution = 0.25

smoke = CenteredGrid(Noise(), extrapolation.BOUNDARY, x=xsize, y= ysize, bounds=Box(x=xsize*resolution, y=ysize*resolution))  # sampled at cell centers
velocity = StaggeredGrid(0, extrapolation.ZERO, x=xsize, y= ysize, bounds=Box(x=xsize*resolution, y=ysize*resolution))


logging.info("scalar component size:")
logging.info(smoke.shape)
logging.info("vector component size:")
logging.info(velocity.shape)


X = []
Y = []

counter = 0

duration = 21
samplingtime = 1.5
history = 2
steps = 0

TrainLen = 15600*history
numdata = 0

NU = 0.01

def step(velocity, smoke, dt=1.5, buoyancy_factor=0.5):
    smoke = advect.semi_lagrangian(smoke, velocity, dt) 
    buoyancy_force = (smoke * (0, buoyancy_factor)).at(velocity)  # resamples smoke to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, dt) + dt * buoyancy_force
    velocity = diffuse.explicit(velocity, NU, dt)
    velocity, _ = fluid.make_incompressible(velocity)
    return velocity, smoke

logging.info("Generating training trajectories...")
while numdata < TrainLen:

  smoke = CenteredGrid(Noise(), extrapolation.BOUNDARY, x=xsize, y= ysize, bounds=Box(x=xsize*resolution, y=ysize*resolution))  # sampled at cell centers
  velocity = StaggeredGrid(0, extrapolation.ZERO, x=xsize, y= ysize, bounds=Box(x=xsize*resolution, y=ysize*resolution))


  #wait for 4 samples before actually collecting data, to avoid having purely noisy scalar field and all-0 vector field
  for j in range(3):
    step(velocity, smoke)


  for i in range(3):
    velocity, smoke = step(velocity, smoke)

    scalarpart = smoke.values.native(order = smoke.shape)
    vectorpart = velocity.staggered_tensor().native('x,y,vector')[1:,1:,:]

    multivector = np.stack((scalarpart, vectorpart[:,:,0], vectorpart[:,:,1]))


    if steps != history:
      X.append(multivector)
      numdata = len(X)
      steps +=1
    else:
      Y.append(multivector)
      steps = 0


    if steps == 0:
      X = np.asarray(X)
      Y = np.asarray(Y)

      print(X.shape)
      print(Y.shape)

      logging.info(f"processed: {numdata} / {TrainLen}")

      np.save(PATH + 'X_trajectories.npy', X)
      np.save(PATH +'Y_trajectories.npy', Y)
      
      X = list(X)
      Y = list(Y)