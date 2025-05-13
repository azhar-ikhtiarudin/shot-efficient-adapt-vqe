# Shot Reduction in ADAPT-VQE via Reused Pauli Measurements and Variance-Based Shot Allocation

## Introduction
This repository implements the Shot Optimized ADAPT-VQE protocol. It reduces the number of shots required for variational quantum eigensolver (VQE) simulations by reusing Pauli measurements and utilizing variance-based shot allocation. This technique optimizes computational resources and enhances the efficiency of quantum simulations.

## How to run the code
This code sets up an environment using Python 3.12, along with key libraries such as Qiskit 1.0.2 for quantum simulations, OpenFermion 1.6.1 for Hamiltonian and fermionic manipulation, and PySCF 2.6.0 for quantum chemistry tasks. 

For an easy setup, you can use the provided `shot-optimized-adapt-vqe.yml` file to install the required dependencies by running the following command:

```bash
conda env create -f shot-optimized-adapt-vqe.yml
```
This will install all the required dependencies and set up the environment. Once the environment is created, activate it using:

```bash
conda activate shot-optimized-adapt-vqe
```


## Example Scripts
There are several example case in this repository, 

## Code References

The code structure for the ADAPT-VQE implementation in this repository is based on [Ramoa et al.'s code](https://github.com/mafaldaramoa/ceo-adapt-vqe), extended to include shot-based calculations using [Qiskit Sampler and Estimator](https://github.com/Qiskit/qiskit), as well as additional methods developed in this work.

The on-the-fly variance calculation was adapted from [Leung et al.'s code](https://github.com/LeungSamWai/OptimizingMeasurement), with modifications to generalize it for an arbitrary N-qubit system and to integrate it with the ADAPT-VQE protocol.

For detailed publication references, see the bibliography in the [main paper](https://#).
