# MMTMTONN
### [Multi–materials topology optimization using deep neural network for coupled thermo–mechanical problems](https://doi.org/10.1016/j.compstruc.2023.107218)

Md. Imrul Reza Shishir, Alireza Tabarraei

Multiscale Material Modelding Labratory, Department of Mechanical Engineering and Engineering Science, The University of North Carolina at Charlotte, Charlotte, NC 28223, USA

# Abstract
In this paper, a density-based topology optimization method using neural networks is used to design domains composed of multi-materials under combined thermo–mechanical loading. The neural network produces the topology density field by minimizing a loss function defined for the problem. Length scale control is conducted using a Fourier space projection. JAX, a high-performance automatic differentiation (AD) Python library is used to build an end-to-end differentiable neural network. The model can handle the sensitivity analysis automatically using the backpropagation process (automatic differentiation of loss function) and the need for solving adjoint equations manual sensitivity analysis is removed. The optimization problem is solved by minimizing the loss function of the network and other optimization algorithms such as the Method of Moving Asymptotes (MMA) are not required. The performance of the method is demonstrated through several examples and compared with those obtained from the popular Solid Isotropic Material with Penalization (SIMP) method. The network is able to handle high-resolution re-sampling, resulting in a more refined and smooth variation of optimal topologies.
