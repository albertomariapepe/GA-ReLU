# GA-ReLU
submitted to ICLR Worshop on AI4DifferentialEquations in Science

# Requirements

```
pip install phiflow
pip install cliffordlayers
```

# How to run

First you need to generate simulation data. You can do so via

```
pyton3 generatedata.py
pyton3 generatedataval.py
pyton3 generatedatatest.py
```

# Example usage:

```
python3 training.py -m CF -A 3 -b 32 -s 28996 -t 15600 -p 10
```

this line will launch the training a Clifford FNO, with the 3rd activation function in act.py (our GA-ReLU), using batch size = 32, random seed = 28996, using 15600 training examples and implementing early stopping with a patience of 10. Alternatively, you can write multiple lines with different parameters and simply launch

```
python3 run.py
```


