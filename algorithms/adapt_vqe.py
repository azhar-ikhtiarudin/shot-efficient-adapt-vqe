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

        ### Pauli Decomposition and Reusing

        self.saved_measurement = {}
        self.full_gradient_data = []

        # Init Gradient Measurement and Savings
        self.gradient_list = []
        pauli_list = PauliList(['I'*self.n])
        coeff_list = np.array([])
        self.operator_list = []

        for i in range(len(self.pool.operators)):
            gradient = commutator(self.qubit_hamiltonian, self.pool.operators[i].q_operator)
            gradient_qiskit = to_qiskit_operator(gradient)
            
            print(f"\nPool-{i}:", gradient_qiskit.paulis)
            
            self.gradient_list.append(gradient_qiskit)
            self.operator_list.append(to_qiskit_operator(self.pool.operators[i].q_operator))
            
            pauli_list = pauli_list.insert(len(pauli_list), gradient_qiskit._pauli_list)
            coeff_list = np.concatenate((coeff_list, gradient_qiskit.coeffs))

        pauli_list = pauli_list.delete(0)
        self.gradient_obs_list = SparsePauliOp(pauli_list, coeff_list)
        
        # Gradient and Hamiltonian Observable
        print("\nGradient observable List:", self.gradient_obs_list.paulis)
        print("Energy Hamiltonian:", self.qiskit_hamiltonian.paulis)

        self.commuted_gradient_obs_list = self.gradient_obs_list.group_commuting(qubit_wise=True)


        Hqis_c = self.qiskit_hamiltonian.group_commuting(qubit_wise=True)
        Hqis_c_array = []
        for i in range(len(Hqis_c)):
            Hqis_c_array.append(Hqis_c[i].paulis[0])
        
        print("Hamiltonian Commuted List:", Hqis_c_array)


        for g in self.commuted_gradient_obs_list:
            print("Loop through commuted gradient:", g.paulis)

            for h in Hqis_c_array:
                is_commute = False
                print(f"\t\t> Compare {g[0].paulis} with {h}")
                is_commute = is_commute | self.is_qubitwise_commuting(str(g[0].paulis[0]), str(h))
                print(f'\t\t  Is Commute? {is_commute}')
                
                if is_commute:
                    print("\t\t\tCommuted ✅")
                    self.saved_measurement[str(h)] = 0
                    break



    def is_qubitwise_commuting(self, pauli1: str, pauli2: str) -> bool:
        """
        Check if two Pauli strings are qubit-wise commuting.
        :param pauli1: First Pauli string (e.g., "IXYZ").
        :param pauli2: Second Pauli string (e.g., "XZIZ").
        :return: True if they are qubit-wise commuting, False otherwise.
        """
        if len(pauli1) != len(pauli2):
            raise ValueError("Pauli strings must have the same length.")
        
        commutingradientpairs = {('I', 'Z'), ('I', 'I'),
                        ('X', 'X'), ('Y', 'Y'), ('Z', 'Z')}
        
        for p1, p2 in zip(pauli1, pauli2):
            if (p1, p2) not in commutingradientpairs and (p2, p1) not in commutingradientpairs:
                return False
        
        return True

    def run(self):
        if self.vrb: print("\n. . . ======= Start Run ADAPT-VQE ======= . . .")
        self.initialize()

        finished = False
        while not finished and self.data.iteration_counter < self.max_adapt_iter:
            finished = self.run_iteration()
        
        if not finished:
            viable_candidates, viable_gradients, total_norm, max_norm = (
                self.rank_gradients()
            )
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
            self.save_to_json(f'data_{self.molecule_name}_N={self.shots_budget}_k={self.k}_Nexp={self.N_experiments}_T={formatted_end_time}_{self.backend_type}_{self.noise_level}.json')

        else:
            print("\n. . . ======= Maximum iteration reached before converged! ======= . . . \n")
            self.save_to_json(f'data_{self.molecule_name}_N={self.shots_budget}_k={self.k}_Nexp={self.N_experiments}_T={formatted_end_time}_{self.backend_type}_{self.noise_level}.json')
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
        
        with open(filepath, 'w') as json_file:
            json.dump(self.full_gradient_data, json_file)

        print(f"Data saved to {filepath}")
        

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

        print(self.energy_statevector)


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

        print("\n\n# Before Energy Optimization")
        for i in range(len(self.data.evolution.its_data)):
            print(self.data.evolution.its_data[i].energies_vpsr)
            print(self.data.evolution.its_data[i].shots_vpsr)
        
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


        print("\n\n# Before Complete Iteration")
        for i in range(len(self.data.evolution.its_data)):
            print(self.data.evolution.its_data[i].energies_vpsr)
            print(self.data.evolution.its_data[i].shots_vpsr)

        self.complete_iteration(energy, total_norm, self.iteration_sel_gradients)

        return finished

    
    def start_iteration(self):
        
        if self.vrb: print(f"\n. . . ======= ADAPT-VQE Iteration {self.data.iteration_counter + 1} ======= . . .")
        

        viable_candidates, viable_gradients, total_norm, max_norm = ( 
            self.rank_gradients() 
        )

        print("\n# Rank Rank Total Results (Recycling): ")

        viable_candidates, viable_gradients, total_norm, max_norm = (
            self.rank_gradients_recycling()
        )

        print("\n# Rank Gradients Shots Allocation: ")

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
    
    def rank_gradients_shots_based(self, coefficients=None, indices=None):

        gradient_list = []
        pauli_list = PauliList(['IIII'])
        coeff_list = np.array([])
        operator_list = []

        # OBTAIN GRADIENT OBSERVABLE AND GROUPING
        for i in range(len(self.pool.operators)):
            # print(f"\nPool-{i}")
            gradient = commutator(self.qubit_hamiltonian, self.pool.operators[i].q_operator)
            gradient_qiskit = to_qiskit_operator(gradient)
            gradient_list.append(gradient_qiskit)
            operator_list.append(to_qiskit_operator(self.pool.operators[i].q_operator))
            
            pauli_list = pauli_list.insert(len(pauli_list), gradient_qiskit._pauli_list)
            coeff_list = np.concatenate((coeff_list, gradient_qiskit.coeffs))

        pauli_list = pauli_list.delete(0)

        
        # GRADIENT OBSERVABLE 
        gradient_obs_list = SparsePauliOp(pauli_list, coeff_list)
        self.commuted_gradient_obs_list = gradient_obs_list.group_commuting(qubit_wise=True)

        print("\nList of Observables")
        
        print(f"# Hamiltonian H")
        print(self.commuted_hamiltonian)
        
        print(f"# Operator A")
        print(operator_list)

        print(f"# Gradient [H,A]")
        print(self.commuted_gradient_obs_list)

        # QUANTUM MEASUREMENT

        if indices is None or coefficients is None:
            ansatz = self.ref_circuit
        else:
            parameters = ParameterVector("theta", len(indices))
            ansatz = self.pool.get_parameterized_circuit(indices, coefficients, parameters)
            ansatz = self.ref_circuit.compose(ansatz)

        print("\nAnsatz for Gradient:")
        print(ansatz)
        
        # SHOTS ALLOCATION
        SHOTS_GRAD = 1024
        self.shots_budget_grad = SHOTS_GRAD*len(self.commuted_gradient_obs_list)
        
        shots_uniform = self.uniform_shots_distribution(self.shots_budget_grad, len(self.commuted_gradient_obs_list))
        shots_vpsr = self.variance_shots_distribution_gradient(self.shots_budget_grad, self.k, coefficients, ansatz, type='vpsr')
        shots_vmsa = self.variance_shots_distribution_gradient(self.shots_budget_grad, self.k, coefficients, ansatz, type='vmsa')
        
        gradient_result_uniform_list = []
        gradient_result_vpsr_list = []
        gradient_result_vmsa_list = []
        self.N_experiments_grad = 5

        for _ in range(self.N_experiments_grad):
            gradient_result_uniform = self.calculate_gradient_result_sampler(coefficients, ansatz, gradient_list, shots_uniform)
            gradient_result_vpsr = self.calculate_gradient_result_sampler(coefficients, ansatz, gradient_list, shots_vpsr)
            gradient_result_vmsa = self.calculate_gradient_result_sampler(coefficients, ansatz, gradient_list, shots_vmsa)

            gradient_result_uniform_list.append(gradient_result_uniform)
            gradient_result_vpsr_list.append(gradient_result_vpsr)
            gradient_result_vmsa_list.append(gradient_result_vmsa)
        
        single_iter_gradient_data = {
            'iter':self.data.iteration_counter,
            'uniform':gradient_result_uniform_list,
            'vpsr':gradient_result_vpsr_list,
            'vmsa':gradient_result_vmsa_list,
            'shots_uniform':shots_uniform,
            'shots_vpsr':shots_vpsr,
            'shots_vmsa':shots_vmsa
        }

        self.full_gradient_data.append(single_iter_gradient_data)

    def calculate_gradient_result_sampler(self, coefficients, ansatz, gradient_list, shots):
        gradient_value = {}
        coeff_value = {}
        
        for clique_idx, cliques in enumerate(self.commuted_gradient_obs_list):

            ansatz_clique = ansatz.copy()
            for j, pauli in enumerate(cliques[0].paulis[0]):
                if (pauli == self.PauliY):
                    ansatz_clique.sdg(j)
                    ansatz_clique.h(j)
                elif (pauli == self.PauliX):
                    ansatz_clique.h(j)
            
            ansatz_clique.measure_all()
            
            job = self.sampler.run(pubs=[(ansatz_clique, coefficients)], shots=shots[clique_idx])
            counts = job.result()[0].data.meas.get_counts()
            probs = self.get_probability_distribution(counts, shots[clique_idx], self.n)

            for commuted_term_idx in range(len(self.commuted_gradient_obs_list[clique_idx])):

                pauli_string = self.commuted_gradient_obs_list[clique_idx][commuted_term_idx].paulis[0].to_label()
                pauli_coeffs = self.commuted_gradient_obs_list[clique_idx][commuted_term_idx].coeffs[0].real
                
                eigen_value = self.get_eigenvalues(pauli_string)
                res = np.dot(eigen_value, probs)

                gradient_value[pauli_string] = res
                coeff_value[pauli_string] = pauli_coeffs
        
        # CALCULATE GRADIENT FROM MEASUREMENT RESULTS

        gradient_result_list = []
        for gradient in gradient_list:

            gradient_result = 0
            for pauli in gradient.paulis:
                res = gradient_value[str(pauli)] * coeff_value[str(pauli)]
                # print(res)
                gradient_result += res
            
            gradient_result_list.append(gradient_result)

        return gradient_result_list


    def rank_gradients_recycling(self):
        
        # Rank Gradients
        sel_gradients = []
        sel_indices = []
        total_norm = 0

        for index, gradient in enumerate(self.gradient_list):
            print(f"\n\tGradient-{index}")

            # Calculate for each Pauli Strings in each Gradient
            gradient_val = 0
            self.pool.imp_type = ImplementationType.SPARSE

            for pauli_string in gradient:
                pauli = str(pauli_string.paulis[0])
                coeffs = pauli_string.coeffs[0]

                if pauli in self.saved_measurement:
                    val = self.saved_measurement[pauli] * coeffs
                    print(f"\t\tGradient used: {pauli} Val:{val.real} ✨")
                    gradient_val += val

                else:
                    val = self.single_pauli_measure(pauli, self.coefficients, self.indices) * coeffs
                    print(f"\t\tPerforming Quantum Measurement for {pauli}: {val.real}")
                    gradient_val += val
            
            if self.vrb: print(f"\tvalue = {gradient_val}")
            
            if np.abs(gradient_val) < 10**-8:
                continue

            sel_gradients, sel_indices = self.place_gradient(
                gradient_val, index, sel_gradients, sel_indices
            )

            if index not in self.pool.parent_range:
                total_norm += gradient_val**2
                print(f"\ttotal norm = {total_norm} ✅")

        
        # for gradient in self.commuted_gradient_obs_list:
        #     print(gradient)
        
        # breakpoint()

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


    def single_pauli_measure(self, pauli, coefficients=None, indices=None):
        observable = SparsePauliOp([pauli])

        if indices is None or coefficients is None: 
            ansatz = self.ref_circuit
            pub = (ansatz, [observable])

        else:
            parameters = ParameterVector("theta", len(self.indices))
            ansatz_initial = self.pool.get_parameterized_circuit(indices, coefficients, parameters)
            ansatz_initial = self.ref_circuit.compose(ansatz_initial)
            ansatz = ansatz_initial
        
            pub = (ansatz, [observable], [coefficients])
        
        result = self.estimator.run(pubs=[pub]).result()
        single_pauli_energy = result[0].data.evs[0]

        # Using Linear Algebra
        observable = SparsePauliOp(pauli[::-1])
        sparse_obs = observable.to_matrix(sparse=True)
        
        ket = self.get_state(coefficients, indices, self.sparse_ref_state) 
        bra = ket.transpose().conj()

        exp_val = bra.dot(sparse_obs.dot(ket))[0,0].real

        return exp_val
        # return single_pauli_energy


    def rank_gradients(self, coefficients=None, indices=None):
        
        print(f"\n # Rank Gradients (Pool size = {self.pool.size})")

        sel_gradients = []
        sel_indices = []
        total_norm = 0

        for index in range(self.pool.size):

            if self.vrb: print("\n\t# === Evaluating Gradient === ", index)

            gradient = self.eval_candidate_gradient(index, coefficients, indices)
            
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
        operator_sparse = get_sparse_operator(operator, self.n)

        observable_sparse = 2 * self.qubit_hamiltonian_sparse @ operator_sparse

        ket = self.get_state(self.coefficients, self.indices, self.sparse_ref_state)        
        bra = ket.transpose().conj()
        gradient = (bra.dot(observable_sparse.dot(ket)))[0,0].real

        return gradient

    def get_state(self, coefficients=None, indices=None, ref_state=None):

        state = self.sparse_ref_state
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
        # print("\n # Grow and Update Ansatz")
        
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


        print("Before Process Iteration")
        for i in range(len(self.data.evolution.its_data)):
            print(self.data.evolution.its_data[i].energies_vpsr)
            print(self.data.evolution.its_data[i].shots_vpsr)

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

        print("After Process Iteration")
        for i in range(len(self.data.evolution.its_data)):
            print(self.data.evolution.its_data[i].energies_vpsr)
            print(self.data.evolution.its_data[i].shots_vpsr)

        # Update current state
        print("\n # Complete Iteration")
        print("\tCurrent energy:", self.energy, "change of", energy_change)
        print(f"\tCurrent ansatz: {list(self.indices)}")

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

        print("\nCoefficients and Indices")
        print(f"\n\tError Percentage: {(self.exact_energy - opt_energy)/self.exact_energy*100}")
        print("\tself.coefficients initial:", self.coefficients)
        print("\tself.indices:", self.indices)

        return opt_energy
    
    
    def evaluate_energy(self, coefficients=None, indices=None):

        print(f"\n\t> Opt Iteration-{self.cost_history_dict['iters']}")


        ### ENERGY WITH GRADIENT RECYCLING
        energy_val = 0
        for index, pauli_string in enumerate(self.qiskit_hamiltonian):
            pauli = str(pauli_string.paulis[0])
            coeffs = pauli_string.coeffs[0]
            val = self.single_pauli_measure(pauli, coefficients, indices)
            print(index, pauli, val)
            
            if pauli in self.saved_measurement:
                self.saved_measurement[pauli] = val
        
            energy_val += val * coeffs

        print(energy_val)
        print(self.saved_measurement)
        # breakpoint()


        ### QISKIT ESTIMATOR DEFAULT
        # self.qiskit_hamiltonian = to_qiskit_operator(self.qubit_hamiltonian)

        if indices is None or coefficients is None: 
            ansatz = self.ref_circuit
            hamiltonian = self.qiskit_hamiltonian
            pub = (ansatz, [hamiltonian])

        else:
            parameters = ParameterVector("theta", len(indices))
            ansatz_initial = self.pool.get_parameterized_circuit(indices, coefficients, parameters)
            ansatz_initial = self.ref_circuit.compose(ansatz_initial)
            ansatz = ansatz_initial
            hamiltonian = self.qiskit_hamiltonian
        
            pub = (ansatz, [hamiltonian], [coefficients])
        
        print("\n\t# === Evaluating Energy === ")
        print("Coefficients:", coefficients)
        print("Indices:", indices)
        ket = self.get_state(coefficients, indices, self.sparse_ref_state) 
        print("Ket during VQE Parameter Optimization:", ket)

        print('\n\nAnsatz for Estimator:', ansatz)

        result = self.estimator.run(pubs=[pub]).result()
              
        energy_qiskit_estimator = result[0].data.evs[0]
        
        print("\n\t>> Qiskit Estimator Energy Evaluation")
        print(f"\t\tenergy_qiskit_estimator: {energy_qiskit_estimator} mHa,   c.a.e = {np.abs(energy_qiskit_estimator-self.exact_energy)*627.5094} kcal/mol")
        # breakpoint()
        # energy_qiskit_estimator = energy_val

        ##### QISKIT SAMPLER

        print(f"\n\t>> Qiskit Sampler Energy Evaluation ")
        if indices is None or coefficients is None:
            ansatz = self.ref_circuit
        else:
            parameters = ParameterVector("theta", len(indices))
            ansatz = self.pool.get_parameterized_circuit(indices, coefficients, parameters)
            ansatz = self.ref_circuit.compose(ansatz)

        # print('\n\nAnsatz for Sampler:', ansatz)
        shots_uniform = self.uniform_shots_distribution(self.shots_budget, len(self.commuted_hamiltonian))
        shots_vpsr = self.variance_shots_distribution(self.shots_budget, self.k, coefficients, ansatz, type='vpsr')
        shots_vmsa = self.variance_shots_distribution(self.shots_budget, self.k, coefficients, ansatz, type='vmsa')

        energy_uniform_list = []
        energy_vpsr_list = []
        energy_vmsa_list = []

        for _ in range(self.N_experiments):
            energy_uniform = self.calculate_exp_value_sampler(coefficients, ansatz, shots_uniform)
            energy_vpsr = self.calculate_exp_value_sampler(coefficients, ansatz, shots_vpsr)
            energy_vmsa = self.calculate_exp_value_sampler(coefficients, ansatz, shots_vmsa)
            energy_uniform_list.append(energy_uniform)
            energy_vpsr_list.append(energy_vpsr)
            energy_vmsa_list.append(energy_vmsa)

        chemac = 627.5094
        energy_uniform = np.mean(energy_uniform_list)
        energy_vpsr = np.mean(energy_vpsr_list)
        energy_vmsa = np.mean(energy_vmsa_list)

        std_uniform = np.std(energy_uniform_list)
        std_vpsr = np.std(energy_vpsr_list)
        std_vmsa = np.std(energy_vmsa_list)
        
        # print("Energy Uniform:", energy_uniform_list)
        # print("Energy VMSA:", energy_vmsa_list)
        # print("Energy VPSR:", energy_vpsr_list)
        
        print("\t\tShots Uniform:", shots_uniform, "->", np.sum(shots_uniform))
        print("\t\tShots VMSA:", shots_vmsa, "->", np.sum(shots_vmsa)+len(self.commuted_hamiltonian)*self.k)
        print("\t\tShots VPSR:", shots_vpsr, "->", np.sum(shots_vpsr)+len(self.commuted_hamiltonian)*self.k)
        
        print("\t\tEnergy Uniform:", energy_uniform, "Error=", np.abs(energy_uniform-self.exact_energy)*chemac, "STD =", std_uniform)
        print("\t\tEnergy VMSA:", energy_vmsa, "Error=", np.abs(energy_vmsa-self.exact_energy)*chemac, "STD =", std_vmsa)
        print("\t\tEnergy VPSR:", energy_vpsr, "Error=", np.abs(energy_vpsr-self.exact_energy)*chemac, "STD =", std_vpsr)

        self.cost_history_dict['iters'] += 1
        self.cost_history_dict['previous_vector'] = coefficients
        self.cost_history_dict['cost_history'].append(energy_qiskit_estimator)

        self.energies_statevector.append(energy_qiskit_estimator)
        self.energies_uniform.append(energy_uniform)
        self.energies_vmsa.append(energy_vmsa)
        self.energies_vpsr.append(energy_vpsr)
        self.std_uniform.append(std_uniform)
        self.std_vmsa.append(std_vmsa)
        self.std_vpsr.append(std_vpsr)
        self.shots_uniform.append(shots_uniform)
        self.shots_vmsa.append(shots_vmsa)
        self.shots_vpsr.append(shots_vpsr)

        error_chemac = np.abs(energy_qiskit_estimator - self.exact_energy) * 627.5094
        if error_chemac > 1:
            self.shots_chemac += np.sum(shots_vpsr)
        print(f"\t\tAccumulated shots up to c.a.e: {self.shots_chemac} -> recent: {np.sum(shots_vpsr)} {shots_vpsr}")
  
        return energy_qiskit_estimator
        # return energy_qiskit_sampler

    def calculate_exp_value_sampler(self, coefficients, ansatz, shots):

        ansatz_cliques = []
        energy_qiskit_sampler = 0.0
        
        for i, cliques in enumerate(self.commuted_hamiltonian):

            ansatz_clique = ansatz.copy()
            for j, pauli in enumerate(cliques[0].paulis[0]):
                if (pauli == self.PauliY):
                    ansatz_clique.sdg(j)
                    ansatz_clique.h(j)
                elif (pauli == self.PauliX):
                    ansatz_clique.h(j)

            ansatz_clique.measure_all()

            ansatz_cliques.append(ansatz_clique)

            job = self.sampler.run(pubs=[(ansatz_clique, coefficients)], shots = shots[i])

            counts = job.result()[0].data.meas.get_counts()
            # print("Counts:", counts)

            probs = self.get_probability_distribution(counts, shots[i], self.n)

            for pauli_string in cliques:
                eigen_value = self.get_eigenvalues(pauli_string.to_list()[0][0])
                
                res = np.dot(eigen_value, probs) * pauli_string.coeffs
                
                energy_qiskit_sampler += res[0].real
            
        return energy_qiskit_sampler
    


    def uniform_shots_distribution(self, N, l):
        shots = [ N // l ] * l
        for i in range(N % l): shots[i] += 1
        return shots
    
    def variance_shots_distribution_gradient(self, N, k, coefficients, ansatz, type):

        ansatz_cliques = []

        std_cliques = []
        for i, cliques in enumerate(self.commuted_gradient_obs_list):
            # print(cliques)
            ansatz_clique = ansatz.copy()
            for j, pauli in enumerate(cliques[0].paulis[0]):
                if (pauli == self.PauliY):
                    ansatz_clique.sdg(j)
                    ansatz_clique.h(j)
                elif (pauli == self.PauliX):
                    ansatz_clique.h(j)

            ansatz_clique.measure_all()
            ansatz_cliques.append(ansatz_clique)
            # print("Ansatz Clique ISA", ansatz_clique)

            job = self.sampler.run(pubs=[(ansatz_clique, coefficients)], shots = k)

            bitstrings = job.result()[0].data.meas.get_bitstrings()
            # print("Bitstirings:",bitstrings)

            results_array = self.convert_bitstrings_to_arrays(bitstrings, self.n)

            results_one_clique = []
            for m, count_res in enumerate(results_array):
                # print(f"\nResults of shot-{m+1}")
                exp_pauli_clique = []
                for pauli_string in cliques:
                    eigen_value = self.get_eigenvalues(pauli_string.to_list()[0][0])
                    # print(eigen_value)
                    # print(count_res)
                    res = np.dot(eigen_value, count_res) * pauli_string.coeffs
                    exp_pauli_clique.append(res[0].real)
                results_one_clique.append(np.sum(exp_pauli_clique))
            
            # print(f"\nResults of Clique-{i}", results_one_clique)
            # print(f"\nSTD of Clique-{i}", np.std(results_one_clique))
            std_cliques.append(np.std(results_one_clique))

        if sum(std_cliques) == 0:
            ratio_for_theta = [1/3 for _ in std_cliques]
        else:
            ratio_for_theta = [ v/sum(std_cliques) for v in std_cliques]
        
        # print("\t\tRatio for Theta", ratio_for_theta)


        # Shots Assignment Equations
        if type == 'vmsa':
            print("k", k)
            print("std cliques", len(std_cliques))
            new_shots_budget = (self.shots_budget_grad - k*len(std_cliques))
        elif type == 'vpsr':
            new_shots_budget = (self.shots_budget_grad - k*len(std_cliques))*sum(ratio_for_theta)**2/len(std_cliques)/sum([v**2 for v in ratio_for_theta])
        
        # print("\t\tNew Shots budget:",new_shots_budget)
        new_shots = [max(1, round(new_shots_budget * ratio_for_theta[i])) for i in range(len(std_cliques))]

        return new_shots
    
    def variance_shots_distribution(self, N, k, coefficients, ansatz, type):

        ansatz_cliques = []

        std_cliques = []
        for i, cliques in enumerate(self.commuted_hamiltonian):
            # print(cliques)
            ansatz_clique = ansatz.copy()
            for j, pauli in enumerate(cliques[0].paulis[0]):
                if (pauli == self.PauliY):
                    ansatz_clique.sdg(j)
                    ansatz_clique.h(j)
                elif (pauli == self.PauliX):
                    ansatz_clique.h(j)

            ansatz_clique.measure_all()
            ansatz_cliques.append(ansatz_clique)
            # print("Ansatz Clique ISA", ansatz_clique)

            job = self.sampler.run(pubs=[(ansatz_clique, coefficients)], shots = k)

            bitstrings = job.result()[0].data.meas.get_bitstrings()
            # print("Bitstirings:",bitstrings)

            results_array = self.convert_bitstrings_to_arrays(bitstrings, self.n)

            results_one_clique = []
            for m, count_res in enumerate(results_array):
                # print(f"\nResults of shot-{m+1}")
                exp_pauli_clique = []
                for pauli_string in cliques:
                    eigen_value = self.get_eigenvalues(pauli_string.to_list()[0][0])
                    # print(eigen_value)
                    # print(count_res)
                    res = np.dot(eigen_value, count_res) * pauli_string.coeffs
                    exp_pauli_clique.append(res[0].real)
                results_one_clique.append(np.sum(exp_pauli_clique))
            
            # print(f"\nResults of Clique-{i}", results_one_clique)
            # print(f"\nSTD of Clique-{i}", np.std(results_one_clique))
            std_cliques.append(np.std(results_one_clique))

        if sum(std_cliques) == 0:
            ratio_for_theta = [1/3 for _ in std_cliques]
        else:
            ratio_for_theta = [ v/sum(std_cliques) for v in std_cliques]
        
        # print("\t\tRatio for Theta", ratio_for_theta)


        # Shots Assignment Equations
        if type == 'vmsa':
            # print("k", k)
            # print("std cliques", len(std_cliques))
            new_shots_budget = (self.shots_budget - k*len(std_cliques))
        elif type == 'vpsr':
            new_shots_budget = (self.shots_budget - k*len(std_cliques))*sum(ratio_for_theta)**2/len(std_cliques)/sum([v**2 for v in ratio_for_theta])
        
        # print("\t\tNew Shots budget:",new_shots_budget)
        new_shots = [max(1, round(new_shots_budget * ratio_for_theta[i])) for i in range(len(std_cliques))]

        return new_shots
    
    
    def convert_bitstrings_to_arrays(self, bitstrings, N):
        all_possible_outcomes = [''.join(format(i, '0' + str(N) + 'b')) for i in range(2**N)]
        outcome_to_index = {outcome: idx for idx, outcome in enumerate(all_possible_outcomes)}
        # Convert each bitstring to a result array
        results = []
        for bitstring in bitstrings:
            result_array = [0] * (2**N)
            if bitstring in outcome_to_index:
                result_array[outcome_to_index[bitstring]] = 1
            results.append(result_array)

        return results
        

    def get_probability_distribution(self, counts, NUM_SHOTS, N):
        # Generate all possible N-qubit measurement outcomes
        all_possible_outcomes = [''.join(format(i, '0' + str(N) + 'b')) for i in range(2**N)]
        # Ensure all possible outcomes are in counts
        for k in all_possible_outcomes:
            if k not in counts.keys():
                counts[k] = 0
        
        # Sort counts by outcome
        sorted_counts = sorted(counts.items())
        # print("Sorted Counts", sorted_counts)
        
        # Calculate the probability distribution
        output_distr = [v[1] / NUM_SHOTS for v in sorted_counts]
        
        return output_distr
    

    def get_eigenvalues(self, pauli_strings):
        # Define Pauli matrices
        eigen_I = np.array([1, 1])
        eigen_X = np.array([1, -1])
        eigen_Y = np.array([1, -1])
        eigen_Z = np.array([1, -1])

        # Map string characters to Pauli matrices
        pauli_dict = {'I': eigen_I, 'X': eigen_X, 'Y': eigen_Y, 'Z': eigen_Z}

        eigen_vals = 1
        
        for pauli in pauli_strings:
            eigen_vals = np.kron(eigen_vals, pauli_dict[pauli])
        
        return eigen_vals
    

    def compute_state(self, coefficients=None, indices=None, ref_state=None, bra=False):
        if indices is None:
            indices = self.indices
        if coefficients is None:
            coefficients = self.coefficients
        
        if ref_state is None:
            ref_state = self.sparse_ref_state
        state = ref_state.copy()

        if bra:
            coefficients = [-c for c in reversed(coefficients)]
            indices = reversed(indices)
        
        for coefficient, index in zip(coefficients, indices):
            state = self.pool.expm_mult(coefficient, index, state)
        if bra:
            state = state.transpose().conj()
        
        return state
    
    def get_custom_noise_model(self):
        p_reset = self.noise_level
        p_meas = self.noise_level
        p_gate1 = self.noise_level
        p_phase = self.noise_level

        # QuantumError objects
        error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
        error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
        error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
        error_gate2 = error_gate1.tensor(error_gate1)
        error_phase = pauli_error([('Z',p_phase), ('I', 1 - p_phase)])

        # Add errors to noise model
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_reset, "reset")
        noise_model.add_all_qubit_quantum_error(error_meas, "measure")
        noise_model.add_all_qubit_quantum_error(error_gate1.compose(error_phase), ["u1", "u2", "u3"])
        noise_model.add_all_qubit_quantum_error(error_gate2.compose(error_phase.tensor(error_phase)), ["cx"])    

        return noise_model

    
    def get_exact_energy(self, custom_hamiltonian):
        custom_hamiltonian_sparse = get_sparse_operator(custom_hamiltonian)
        eigs, _ = linalg.eigsh(custom_hamiltonian_sparse, k=1, which='SA')
        return eigs[0]
    

    def analyze_k(self):
        print("Analyzing Value of k")
        # N_experiments = 1000
        
        self.initialize()
        coefficients = [-0.11319058]
        indices=[2]

        energy_calculations = []
        for experiment in range(self.N_experiments):
            energy = self.evaluate_energy(coefficients, indices)
            print(f"Experiment {experiment}, Energy = {energy}")
            energy_calculations.append(energy)
        
        print(energy_calculations)

        with open("a.json", "w") as json_file:
            json.dump(energy_calculations, json_file)
        
        self.plot_histogram(energy_calculations, self.exact_energy)
    

    def plot_histogram(self, data, exact_energy):
        plt.hist(data, bins=10, color='blue', alpha=0.7, edgecolor='black', label='Histogram')
        plt.axvline(exact_energy, color='red', linestyle='--', linewidth=1.5, label='Exact Energy')
        for idx, value in enumerate(sorted(data)):
            plt.plot(value, 0, marker='o', color='navy')
        plt.title("Expectation Values Calculations Distribution")
        plt.xlabel("Energy")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def load_and_plot_histogram(self, json_file_path, exact_energy):
        # Load energy calculations from a JSON file
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        print(f"Loaded data from '{json_file_path}': {data}")

        return data