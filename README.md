
# GA-ReLU 🌊  
*Presented at ICLR 2024, Workshop on AI4DifferentialEquations in Science*

## Introduction  
Many differential equations describing physical phenomena are intrinsically geometric in nature 📐. It has been demonstrated that the geometric structure of data can be effectively captured through networks operating in **Geometric Algebra (GA)** using multivectors. These networks show great promise for solving differential equations, yet GA networks remain largely uncharted territory. In this paper, we focus on non-linearities—a challenging aspect when applied to multivectors, as they are typically handled point-wise on each real-valued component. This conventional approach discards the interactions between different elements of the multivector, compromising the inherent geometric structure. To bridge this gap, we propose **GA-ReLU** ⚡, a GA-based approach to the rectified linear unit (ReLU), and demonstrate how it can enhance the solution of **Navier-Stokes** PDEs 🌊.

## Requirements 📦

```
pip install phiflow
pip install cliffordlayers
```

## Getting Started 🚀

First, you need to generate simulation data. You can do so via:

```
python3 generatedata.py
python3 generatedataval.py
python3 generatedatatest.py
```

## Example Usage 🖥️

```
python3 training.py -m CF -A 3 -b 32 -s 28996 -t 15600 -p 10
```

## How to Cite

```
@inproceedings{pepe2024ga,
  title={GA-reLU: an activation function for geometric algebra networks applied to 2d navier-stokes PDEs},
  author={Pepe, Alberto and Buchholz, Sven and Lasenby, Joan},
  booktitle={ICLR 2024 Workshop on AI4DifferentialEquations In Science},
  year={2024}
}
```


This line will launch the training of a Clifford FNO, using the 3rd activation function in `act.py` (our **GA-ReLU**), with batch size = 32, random seed = 28996, 15600 training examples, and early stopping with a patience of 10. Alternatively, you can write multiple lines with different parameters and simply launch:

```
python3 run.py
```
