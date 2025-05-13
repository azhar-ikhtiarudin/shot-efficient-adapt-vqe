# Shot Reduction in ADAPT-VQE via Reused Pauli Measurements and Variance-Based Shot Allocation

## Introduction
This repository contains an implementation of the Shot-Optimized ADAPT-VQE protocol, designed to minimize the number of measurements (shots) needed in Variational Quantum Eigensolver (VQE) simulations. By reusing Pauli measurements and applying variance-based shot allocation, the method improves computational efficiency and optimizes resource usage in quantum simulations.

## How to run the code
This code was originally developed in a Conda environment using Python 3.12, along with key libraries such as Qiskit 1.0.2 for quantum simulations, and OpenFermion 1.6.1 with PySCF 2.6.0 for Hamiltonian construction, fermionic manipulation, and other quantum chemistry tasks.

For an easy setup, you can use the provided `environment.yml` file to install the required dependencies by running the following command:

```bash
conda env create -f environment.yml
```
This will install all the required dependencies and set up the environment. Once the environment is created, activate it using:

```bash
conda activate adapt-vqe-dev
```


## Example Scripts

Several example simulation scripts are provided in this repository, including simulations of H₂ with the Qubit-Excitation (QE) pool and LiH with the Qubit pool. You can run an example using the following command:

```bash
python example_h2_qe_pool.py
```
The provided examples can be customized to use different operator pools or molecules, including custom Hamiltonians. Additionally, a separate script is also provided to analyze the Pauli strings between the Hamiltonian and the gradient observable for H₂.

## Code References

The code structure for the ADAPT-VQE implementation in this repository is based on [Ramoa et al.'s code](https://github.com/mafaldaramoa/ceo-adapt-vqe), extended to include shot-based calculations using [Qiskit Sampler and Estimator](https://github.com/Qiskit/qiskit), as well as additional methods developed in this work.

The on-the-fly variance calculation was adapted from [Leung et al.'s code](https://github.com/LeungSamWai/OptimizingMeasurement), with modifications to generalize it for an arbitrary N-qubit system and to integrate it with the ADAPT-VQE protocol.

For detailed publication references, see the bibliography in the [main paper](https://#).
