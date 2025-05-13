import time
import os
from copy import copy, deepcopy
import numpy as np
from scipy.sparse import csc_matrix, linalg, csr_matrix

from .adapt_data import AdaptData
from src.minimize import minimize_bfgs
from src.utilities import ket_to_vector
from src.circuits import pauli_exp_circuit

from openfermion.transforms import jordan_wigner
from openfermion.utils import commutator
from openfermion.linalg import get_sparse_operator
from openfermion.ops import QubitOperator

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_aer.noise import NoiseModel, pauli_error

from qiskit_algorithms.optimizers import ADAM, SPSA

from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import Pauli, SparsePauliOp, PauliList
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from scipy.optimize import minimize
from src.utilities import to_qiskit_operator

import json
import matplotlib.pyplot as plt


class ImplementationType:
    SPARSE = 0
    QISKIT = 1

class AdaptVQE():
    """
        Main Class for ADAPT-VQE Algorithm
    """

    def __init__(self, pool, molecule, max_adapt_iter, max_opt_iter, 
                    grad_threshold=10**-8, vrb=False, 
                    optimizer_method='bfgs', shots_assignment='vmsa',
                    k=None, shots_budget=1024, seed=None, N_experiments=10, 
                    backend_type='noiseless', custom_hamiltonian=None, noise_level=0):

        self.pool = pool
        self.molecule = molecule
        self.max_adapt_iter = max_adapt_iter
        self.max_opt_iter = max_opt_iter
        self.vrb = vrb
        self.grad_threshold = grad_threshold
        self.data = None
        self.optimizer_method = optimizer_method
        self.shots_assignment = shots_assignment
        self.backend_type = backend_type
        self.noise_level = noise_level
        
        if self.molecule is not None:
            print("Using molecular hamiltonian")
            self.fermionic_hamiltonian = self.molecule.get_molecular_hamiltonian()
            self.qubit_hamiltonian = jordan_wigner(self.fermionic_hamiltonian)
            self.exact_energy = self.molecule.fci_energy
            self.molecule_name = self.molecule.description

        else:
            print("Using Custom Hamiltonian")
            self.qubit_hamiltonian = custom_hamiltonian
            self.molecule_name = 'LiH'
            self.exact_energy = self.get_exact_energy(custom_hamiltonian)
        
        self.qiskit_hamiltonian = to_qiskit_operator(self.qubit_hamiltonian)

        self.n = self.qiskit_hamiltonian.num_qubits
        self.qubit_hamiltonian_sparse = get_sparse_operator(self.qubit_hamiltonian, self.n)
        self.commuted_hamiltonian = self.qiskit_hamiltonian.group_commuting(qubit_wise=True)
        
        print("Exact Energy", self.exact_energy)
        self.window = self.pool.size

        self.k = k
        self.shots_budget = shots_budget * len(self.commuted_hamiltonian)
        print("Shots Budget:", self.shots_budget)
        self.shots_chemac = 0
        self.seed = seed
        self.N_experiments = N_experiments

        if self.backend_type == 'noiseless':
            self.sampler = Sampler()
        elif self.backend_type == 'noisy':
            self.noise_model = self.get_custom_noise_model()
            self.sampler = Sampler(options=dict(backend_options=dict(noise_model=self.noise_model)))

        self.estimator = Estimator()
        self.PauliX = Pauli("X")
        self.PauliZ = Pauli("Z")
        self.PauliI = Pauli("I")
        self.PauliY = Pauli("Y")


        # Hartree Fock Reference State:
        self.ref_determinant = [ 1 for _ in range(self.molecule.n_electrons) ]
        self.ref_determinant += [ 0 for _ in range(self.fermionic_hamiltonian.n_qubits - self.molecule.n_electrons ) ]
        # self.ref_determinant = [ 1 for _ in range(2) ]
        # self.ref_determinant += [ 0 for _ in range(4 - 2) ]
        self.sparse_ref_state = csc_matrix(
            ket_to_vector(self.ref_determinant), dtype=complex
        ).transpose()


        # Reference State Circuit
        self.ref_circuit = QuantumCircuit(self.n)
        for i, qubit in enumerate(self.ref_determinant):
            if qubit == 1 : 
                self.ref_circuit.x(i)
        self.ref_circuit.barrier()

        self.sparse_ref_state = csc_matrix(
            ket_to_vector(self.ref_determinant), dtype = complex
            ).transpose()
        

        self.cost_history_dict = {
            "prev_vector": None,
            "iters":0,
            "cost_history":[],
        }

        self.gradients = np.array(())
        self.iteration_nfevs = []
        self.iteration_ngevs = []
        self.iteration_nits = []
        self.total_norm = 0
        self.sel_gradients = []

        self.energies_statevector = []
        self.energies_uniform = []
        self.energies_vpsr = []
        self.energies_vmsa = []
        self.std_uniform = []
        self.std_vpsr = []
        self.std_vmsa = []
        self.shots_uniform = []
        self.shots_vpsr = []
        self.shots_vmsa = []

        print("Backend Type:", self.backend_type)
        print("Noise Level:", self.noise_level)

    def run(self):
        if self.vrb: print("\n. . . ======= Start Run ADAPT-VQE ======= . . .")
        self.initialize()

        finished = False
        while not finished and self.data.iteration_counter < self.max_adapt_iter:
            finished = self.run_iteration()
        
        if not finished:
            # self.rank_gradients_shots_based(self.coefficients, self.indices)
            # viable_candidates, viable_gradients, total_norm, max_norm = (
            #     self.rank_gradients_recycling()
            # )

            viable_candidates, viable_gradients, total_norm, max_norm = (
                self.rank_gradients()
            )

            # print(viable_candidates, viable_gradients, total_norm)

            if total_norm < self.grad_threshold:
                self.data.close(True) # converge()
                finished = True
            print("Self.Energy", self.energy)
        
        program_end_time = time.localtime()
        formatted_end_time = time.strftime('%d%m%y_%H%M%S', program_end_time)
        
        if finished:
            print("\n. . . ======= Convergence Condition Achieved 🎉🎉🎉 ======= . . .")
            error = self.energy - self.exact_energy
            self.data.shots_chemac = self.shots_chemac
            print(f"\n\t> Energy:")
            print(f"\tFinal Energy = {self.energy}")
            print(f"\tError = {error}")
            print(f"\tError in Chemical accuracy= {error*627.5094} kcal/mol")
            print(f"\tIterations completed = {self.data.iteration_counter}")

            print(f"\n\t> Circuit Property:")
            print(f"\tAnsatz indices = {self.indices}")
            print(f"\tCoefficients = {self.coefficients}")
            print("END TIME: ", formatted_end_time)
            self.save_to_json(f'data_{self.molecule_name}_N={self.shots_budget}_k={self.k}_Nexp={self.N_experiments}_T={formatted_end_time}_{self.backend_type}_{self.noise_level}_{self.pool.name}.json')
            self.save_gradient_to_json(f'data_gradient_{self.molecule_name}_N={self.shots_budget}_k={self.k}_Nexp={self.N_experiments}_T={formatted_end_time}_{self.backend_type}_{self.noise_level}.json')

        else:
            print("\n. . . ======= Maximum iteration reached before converged! ======= . . . \n")
            self.save_to_json(f'data_{self.molecule_name}_N={self.shots_budget}_k={self.k}_Nexp={self.N_experiments}_T={formatted_end_time}_{self.backend_type}_{self.noise_level}_{self.pool.name}.json')
            self.save_gradient_to_json(f'data_gradient_{self.molecule_name}_N={self.shots_budget}_k={self.k}_Nexp={self.N_experiments}_T={formatted_end_time}_{self.backend_type}_{self.noise_level}.json')
            self.data.close(False)
        
        return
    
    def save_gradient_to_json(self, filename):
        print("Save Gradient to JSON")
        
        # Ensure the directory exists
        directory = 'data'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Prepend directory to filename
        filepath = os.path.join(directory, filename)
        
        # with open(filepath, 'w') as json_file:
        #     json.dump(self.full_gradient_data, json_file)

        # print(f"Data saved to {filepath}")
        

    def save_to_json(self, filename):
        print("\n\n# Save to JSON")
        
        # Ensure the directory exists
        directory = 'data'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Prepend directory to filename
        filepath = os.path.join(directory, filename)

        data_list = []
        for i in range(len(self.data.evolution.its_data)):
            data = {
                "energies_statevector": self.data.evolution.its_data[i].energies_statevector,
                "energies_uniform": self.data.evolution.its_data[i].energies_uniform,
                "energies_vmsa": self.data.evolution.its_data[i].energies_vmsa,
                "energies_vpsr": self.data.evolution.its_data[i].energies_vpsr,
                "std_uniform": self.data.evolution.its_data[i].std_uniform,
                "std_vmsa": self.data.evolution.its_data[i].std_vmsa,
                "std_vpsr": self.data.evolution.its_data[i].std_vpsr,
                "shots_uniform": self.data.evolution.its_data[i].shots_uniform,
                "shots_vmsa": self.data.evolution.its_data[i].shots_vmsa,
                "shots_vpsr": self.data.evolution.its_data[i].shots_vpsr,
            }

            data_list.append(data)
        
        result = {
            'pool_name': self.data.pool_name,
            'initial_energy': self.data.initial_energy,
            'fci_energy': self.data.fci_energy,
            'data_list': data_list
        }

        with open(filepath, 'w') as json_file:
            json.dump(result, json_file)

        print(f"Data saved to {filepath}")
        

    def load_data(self, filename):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        self.energy_statevector = data["energy_statevector"]
        self.energy_uniform = data["error_uniform"]
        self.energy_vpsr = data["error_vpsr"]
        self.energy_vmsa = data["error_vmsa"]
        self.std_uniform = data["std_uniform"]
        self.std_vpsr = data["std_vpsr"]
        self.std_vmsa = data["std_vmsa"]

        # print(self.energy_statevector)


    def initialize(self):
        if self.vrb:
            print("\n # Initialize Data ")
        if not self.data:
            self.indices = []
            self.coefficients = []
            self.old_coefficients = []
            self.old_gradients = []

        self.initial_energy = self.evaluate_energy()
        self.energy = self.initial_energy
        print("\n\tInitial Energy = ", self.initial_energy)
        print('\tExact Energt =', self.exact_energy)

        if not self.data: 
            self.data = AdaptData(self.initial_energy, self.pool, self.exact_energy, self.n)

        self.iteration_sel_gradients = None

        self.data.process_initial_iteration(
            self.indices,
            self.energy,
            self.total_norm,
            self.sel_gradients,
            self.coefficients,
            self.gradients,
            self.iteration_nfevs,
            self.iteration_ngevs,
            self.iteration_nits,
            self.energies_statevector,
            self.energies_uniform,
            self.energies_vmsa,
            self.energies_vpsr,
            self.std_uniform,
            self.std_vmsa,
            self.std_vpsr,
            self.shots_uniform,
            self.shots_vmsa,
            self.shots_vpsr
        )

        return


    def run_iteration(self):

        # print("\n\n# Initial Iteration")
        # for i in range(len(self.data.evolution.its_data)):
        #     print(self.data.evolution.its_data[i].energies_vpsr)
        #     print(self.data.evolution.its_data[i].shots_vpsr)

        # Gradient Screening
        finished, viable_candidates, viable_gradients, total_norm = ( 
            self.start_iteration() 
        )

        self.energy_opt_iters = []
        self.shots_iters = []
        
        if finished: 
            return finished

        while viable_candidates:
            energy, gradient, viable_candidates, viable_gradients = self.grow_and_update( 
                viable_candidates, viable_gradients 
            )

        self.energies_statevector = []
        self.energies_uniform = []
        self.energies_vmsa = []
        self.energies_vpsr = []
        self.std_uniform = []
        self.std_vmsa = []
        self.std_vpsr = []
        self.shots_uniform = []
        self.shots_vmsa = []
        self.shots_vpsr = []

        if energy is None: 
            energy = self.optimize(gradient) # Optimize energy with current updated ansatz (additional gradient g)

        self.complete_iteration(energy, total_norm, self.iteration_sel_gradients)

        return finished

    
    def start_iteration(self):
        
        if self.vrb: print(f"\n. . . ======= ADAPT-VQE Iteration {self.data.iteration_counter + 1} ======= . . .")
        
        # print(f"\n # Active Circuit at Adapt iteration {self.data.iteration_counter + 1}:")
        
        # self.rank_gradients_shots_based(self.coefficients, self.indices)

        viable_candidates, viable_gradients, total_norm, max_norm = ( 
            self.rank_gradients() 
        )

        finished = False
        if total_norm < self.grad_threshold:
            self.data.close(True) # converge()
            finished = True
        
        if finished: 
            return finished, viable_candidates, viable_gradients, total_norm

        print(
            f"\tIs Finished? -> {finished}"
        )

        self.iteration_nfevs = []
        self.iteration_ngevs = []
        self.iteration_nits = []
        self.iteration_sel_gradients = []
        self.iteration_qubits = ( set() )

        return finished, viable_candidates, viable_gradients, total_norm
    
    def rank_gradients(self, coefficients=None, indices=None):
        
        print(f"\n # Rank Gradients (Pool size = {self.pool.size})")

        sel_gradients = []
        sel_indices = []
        total_norm = 0

        for index in range(self.pool.size):

            if self.vrb: print("\n\t# === Evaluating Gradient === ", index)
            # print("Coefficients:", coefficients)
            # print("Indices:", indices)
            
            # print("Self.Coefficients:", self.coefficients)
            # print("Self.Indices:", self.indices)

            gradient = self.eval_candidate_gradient(index, coefficients, indices)
            # print(gradient)

            # breakpoint()
            
            if self.vrb: print(f"\t\tvalue = {gradient}")

            if np.abs(gradient) < 10**-8:
                continue

            sel_gradients, sel_indices = self.place_gradient( 
                gradient, index, sel_gradients, sel_indices 
            )

            if index not in self.pool.parent_range: 
                total_norm += gradient**2
                print(f"\t\ttotal norm = {total_norm} ✅")

        total_norm = np.sqrt(total_norm)

        if sel_gradients: 
            max_norm = sel_gradients[0]
        else: 
            max_norm = 0

        if self.vrb:
            print("\n # Gradient Rank Total Results")
            print(f"\n\tTotal gradient norm: {total_norm}")
            print("\tFinal Selected Indices:", sel_indices)
            print("\tFinal Selected Gradients:", sel_gradients)

        return sel_indices, sel_gradients, total_norm, max_norm
    
    
    def eval_candidate_gradient(self, index, coefficients=None, indices=None):

        self.pool.imp_type = ImplementationType.SPARSE

        operator = self.pool.get_q_op(index)
        # print("Operator Q:", operator)
        # operator = self.pool.get_f_op(index)
        # print("Operator F:", operator)
        # breakpoint()
        operator_sparse = get_sparse_operator(operator, self.n)
        observable_sparse = 2 * self.qubit_hamiltonian_sparse @ operator_sparse

        ket = self.get_state(self.coefficients, self.indices, self.sparse_ref_state)        
        bra = ket.transpose().conj()
        gradient = (bra.dot(observable_sparse.dot(ket)))[0,0].real

        return gradient

    def get_state(self, coefficients=None, indices=None, ref_state=None):
        state = self.sparse_ref_state
        print("Get State:", state)
        # breakpoint()
        if coefficients is None or indices is None:
            return state
        else:
            for coefficient, index in zip(coefficients, indices):
                state = self.pool.expm_mult(coefficient, index, state)

        return state

    def place_gradient(self, gradient, index, sel_gradients, sel_indices):

        i = 0

        for sel_gradient in sel_gradients:
            if np.abs(np.abs(gradient) - np.abs(sel_gradient)) < 10**-8:
                condition = self.break_gradient_tie(gradient, sel_gradient)
                if condition: break
            
            elif np.abs(gradient) - np.abs(sel_gradient) >= 10**-8:
                break

            i += 1
        
        if i < self.window:
            sel_indices = sel_indices[:i] + [index] + sel_indices[i : self.window - 1]

            sel_gradients = (
                sel_gradients[:i] + [gradient] + sel_gradients[i : self.window - 1]
            )
        
        return sel_gradients, sel_indices

    def break_gradient_tie(self, gradient, sel_gradient):
        assert np.abs(np.abs(gradient) - np.abs(sel_gradient)) < 10**-8

        condition = np.abs(gradient) > np.abs(sel_gradient)

        return condition
    

    def grow_and_update(self, viable_candidates, viable_gradients):
        
        # Grow Ansatz
        energy, gradient = self.grow_ansatz(viable_candidates, viable_gradients)

        # Update Viable Candidates
        viable_candidates = []

        self.iteration_sel_gradients = np.append(self.iteration_sel_gradients, gradient)
        return energy, gradient, viable_candidates, viable_gradients


    def grow_ansatz(self, viable_candidates, viable_gradients, max_additions=1):

        total_new_nfevs = []
        total_new_ngevs = []
        total_new_nits = []
        gradients = []

        while max_additions > 0:

            new_nfevs = []
            new_ngevs = []
            new_nits = []
            energy = None

            index, gradient = self.find_highest_gradient(viable_candidates, viable_gradients)

            self.indices.append(index)
            self.coefficients = np.append(self.coefficients, 0)
            self.gradients = np.append(self.gradients, gradient)

            if self.data.evolution.indices:
                old_size = len(self.data.evolution.indices[-1])
            else:
                old_size = 0
            new_indices = self.indices[old_size:]

            if new_nfevs:
                total_new_nfevs.append(new_nfevs)
            if new_ngevs:
                total_new_ngevs.append(new_nits)
            if new_nits:
                total_new_nits.append(new_nits)
            
            gradients.append(gradient)
            max_additions -= 1
        
        print("\tOperator(s) added to ansatz:", new_indices)
        self.update_iteration_costs(total_new_nfevs, total_new_ngevs, total_new_nits)

        return energy, gradient

    
    def find_highest_gradient(self, indices, gradients, excluded_range=[]):

        viable_indices = []
        viable_gradients = []
        for index, gradient in zip(indices, gradients):
            if index not in excluded_range:
                viable_indices.append(index)
                viable_gradients.append(gradient)
        
        abs_gradients = [ np.abs(gradient) for gradient in viable_gradients ]
        max_abs_gradient = max(abs_gradients)

        grad_rank = abs_gradients.index(max_abs_gradient)
        index = viable_indices[grad_rank]
        gradient = viable_gradients[grad_rank]

        return index, gradient
    
    def update_iteration_costs(self, new_nfevs=None, new_ngevs=None, new_nits=None):
        if new_nfevs:
            self.iteration_nfevs = self.iteration_nfevs + new_nfevs
        if new_ngevs:
            self.iteration_ngevs = self.iteration_ngevs + new_ngevs
        if new_nits:
            self.iteration_nits = self.iteration_nits + new_nits

    def complete_iteration(self, energy, total_norm=None, sel_gradients=None):

        energy_change = energy - self.energy
        self.energy = energy

        # Save iteration data
        self.data.process_iteration(
            self.indices,
            self.energy,
            total_norm,
            sel_gradients,
            self.coefficients,
            self.gradients,
            self.iteration_nfevs,
            self.iteration_ngevs,
            self.iteration_nits,
            self.energies_statevector,
            self.energies_uniform,
            self.energies_vmsa,
            self.energies_vpsr,
            self.std_uniform,
            self.std_vmsa,
            self.std_vpsr,
            self.shots_uniform,
            self.shots_vmsa,
            self.shots_vpsr
        )

        return    

    def optimize(self, gradient=None):
        """gradient: gradient of the last-added operator"""

        # Full Optimization
        print("\n # Standard VQE Optimization")

        self.cost_history_dict = {
            "prev_vector": None,
            "iters":0,
            "cost_history":[],
        }
        
        initial_coefficients = deepcopy(self.coefficients)
        indices = self.indices.copy()
        g0 = self.gradients
        e0 = self.energy
        maxiters = self.max_opt_iter

        print("\n\tEnergy Optimization Parameter:")
        print("\t\tInitial Coefficients:", initial_coefficients)
        print("\t\tIndices:", indices)
        print("\t\tg0:", g0)
        print("\t\te0:", e0)

        # Scipy Minimize
        res = minimize(
            self.evaluate_energy,
            initial_coefficients,
            args=(indices),
            method=self.optimizer_method,
            # options={"maxiter":1000}
        )

        print("\nScipy Optimize Result:",res)
        
        self.coefficients = res.x
        print("\tself.coefficients updated:", self.coefficients)
        opt_energy = res.fun

        print("\nOptimized Circuit with Coefficients")
        print("Optimization Iteration at ADAPT-VQE Iter:", self.data.iteration_counter,":\n", self.cost_history_dict['cost_history'])

        print(f"\n\tError Percentage: {(self.exact_energy - opt_energy)/self.exact_energy*100}")
        print("\tself.coefficients initial:", self.coefficients)
        print("\tself.indices:", self.indices)

        return opt_energy
    
    
    def evaluate_energy(self, coefficients=None, indices=None):

        print(f"\n\t> Opt Iteration-{self.cost_history_dict['iters']}")


        ket = self.get_state(coefficients, indices, self.sparse_ref_state)
        bra = ket.transpose().conj()
        exp_val = (bra.dot(self.qubit_hamiltonian_sparse.dot(ket)))[0,0].real

        self.cost_history_dict['iters'] += 1
        self.cost_history_dict['previous_vector'] = coefficients
        self.cost_history_dict['cost_history'].append(exp_val)

        self.energies_statevector.append(exp_val)

        return exp_val
