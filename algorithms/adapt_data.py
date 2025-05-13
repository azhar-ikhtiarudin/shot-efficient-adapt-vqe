
import numpy as np
from copy import deepcopy

from src.utilities import cnot_depth, cnot_count, get_qasm
from qiskit import QuantumCircuit

class AnsatzData:

    def __init__(self, coefficients=[], indices=[], sel_gradients=[]):
        self.coefficients = coefficients
        self.indices = indices
        self.sel_gradients = sel_gradients

    def grow(self, indices, new_coefficients, sel_gradients):
        self.indices = indices
        self.coefficients = new_coefficients
        self.sel_gradients = np.append(self.sel_gradients, sel_gradients)

    def remove(self, index, new_coefficients):
        self.indices.pop(index)
        self.coefficients = new_coefficients
        rem_grad = self.sel_gradients.pop(index)

        return rem_grad

    @property
    def size(self):
        return len(self.indices)


class IterationData:
    def __init__(self,
                 ansatz=None,
                 energy=None,
                 error=None,
                 energy_change=None,
                 gradient_norm=None,
                 sel_gradients=None,
                 gradients=None,
                 nfevs=None,
                 ngevs=None,
                 nits=None,
                 energies_statevector=None,
                 energies_uniform=None,
                 energies_vmsa=None,
                 energies_vpsr=None,
                 std_uniform=None,
                 std_vmsa=None,
                 std_vpsr=None,
                 shots_uniform=None,
                 shots_vmsa=None,
                 shots_vpsr=None,
                 ):
        if ansatz:
            self.ansatz = deepcopy(ansatz)
        else:
            self.ansatz = AnsatzData()
        
        self.energy = energy
        self.energy_change = energy_change
        self.error = error
        self.gradient_norm = gradient_norm
        self.sel_gradients = sel_gradients
        self.gradients = gradients
        self.nfevs = nfevs
        self.ngevs = ngevs
        self.nits = nits

        self.energies_statevector = energies_statevector
        self.energies_uniform = energies_uniform
        self.energies_vmsa = energies_vmsa
        self.energies_vpsr = energies_vpsr
        self.std_uniform = std_uniform
        self.std_vmsa = std_vmsa
        self.std_vpsr = std_vpsr
        self.shots_uniform = shots_uniform
        self.shots_vmsa = shots_vmsa
        self.shots_vpsr = shots_vpsr

class EvolutionData:

    def __init__(self, initial_energy, prev_ev_data=None):

        self.initial_energy = initial_energy

        if prev_ev_data:
            self.its_data = prev_ev_data.its_data
        else:
            # List of IterationData objects
            self.its_data = []

    def reg_it(
        self,
        coefficients,
        indices,
        energy,
        error,
        gradient_norm,
        sel_gradients,
        gradients,
        nfevs,
        ngevs,
        nits,
        
        energies_statevector,
        energies_uniform,
        energies_vmsa,
        energies_vpsr,
        std_uniform,
        std_vmsa,
        std_vpsr,
        shots_uniform,
        shots_vmsa,
        shots_vpsr,
    ):

        if self.its_data:
            previous_energy = self.last_it.energy
        else:
            previous_energy = self.initial_energy

        energy_change = energy - previous_energy

        ansatz = deepcopy(self.last_it.ansatz)
        ansatz.grow(indices, coefficients, sel_gradients)

        it_data = IterationData(
            ansatz,
            energy,
            error,
            energy_change,
            gradient_norm,
            sel_gradients,
            gradients,
            nfevs,
            ngevs,
            nits,
            
            energies_statevector,
            energies_uniform,
            energies_vmsa,
            energies_vpsr,
            std_uniform,
            std_vmsa,
            std_vpsr,
            shots_uniform,
            shots_vmsa,
            shots_vpsr,
        )

        self.its_data.append(it_data)

        return

    @property
    def coefficients(self):
        return [it_data.ansatz.coefficients for it_data in self.its_data]

    @property
    def energies(self):
        return [it_data.energy for it_data in self.its_data]

    @property
    def inv_hessians(self):
        return [it_data.inv_hessian for it_data in self.its_data]

    @property
    def gradients(self):
        return [it_data.gradients for it_data in self.its_data]

    @property
    def errors(self):
        return [it_data.error for it_data in self.its_data]

    @property
    def energy_changes(self):
        return [it_data.energy_change for it_data in self.its_data]

    @property
    def gradient_norms(self):
        return [it_data.gradient_norm for it_data in self.its_data]

    @property
    def indices(self):
        return [it_data.ansatz.indices for it_data in self.its_data]

    @property
    def nfevs(self):
        return [it_data.nfevs for it_data in self.its_data]

    @property
    def ngevs(self):
        return [it_data.ngevs for it_data in self.its_data]

    @property
    def nits(self):
        return [it_data.nits for it_data in self.its_data]

    @property
    def sel_gradients(self):
        return [it_data.sel_gradients for it_data in self.its_data]

    @property
    def sizes(self):
        return [len(it_data.ansatz.indices) for it_data in self.its_data]

    @property
    def last_it(self):

        if self.its_data:
            return self.its_data[-1]
        else:
            # No data yet. Return empty IterationData object
            return IterationData()


class AdaptData:
    def __init__(
            self, initial_energy, pool, fci_energy, n, sparse_ref_state=None, file_name=None
    ):
        self.pool_name = pool.name
        self.initial_energy = initial_energy
        self.initial_error = initial_energy - fci_energy
        self.sparse_ref_state = sparse_ref_state

        self.evolution = EvolutionData(initial_energy)
        self.file_name = file_name
        self.iteration_counter = 0
        self.fci_energy = fci_energy
        self.n = n

        self.closed = False
        self.success= False
        self.shots_chemac = 0

    def process_iteration(
            self,
            indices,
            energy,
            gradient_norm,
            selected_gradients,
            coefficients,
            gradients,
            nfevs,
            ngevs,
            nits,
            
            energies_statevector,
            energies_uniform,
            energies_vmsa,
            energies_vpsr,
            std_uniform,
            std_vmsa,
            std_vpsr,
            shots_uniform,
            shots_vmsa,
            shots_vpsr
    ):
        print("\n# Process Iteration")
        error = energy - self.fci_energy
        self.evolution.reg_it(
            coefficients,
            indices,
            energy,
            error,
            gradient_norm,
            selected_gradients,
            gradients,
            nfevs,
            ngevs,
            nits,
            
            energies_statevector,
            energies_uniform,
            energies_vmsa,
            energies_vpsr,
            std_uniform,
            std_vmsa,
            std_vpsr,
            shots_uniform,
            shots_vmsa,
            shots_vpsr
        )

        self.iteration_counter += 1

        return energy
    
    def process_initial_iteration(
            self,
            indices,
            energy,
            gradient_norm,
            selected_gradients,
            coefficients,
            gradients,
            nfevs,
            ngevs,
            nits,
            
            energies_statevector,
            energies_uniform,
            energies_vmsa,
            energies_vpsr,
            std_uniform,
            std_vmsa,
            std_vpsr,
            shots_uniform,
            shots_vmsa,
            shots_vpsr
    ):
        error = energy - self.fci_energy
        # self.evolution.reg_it(
        #     coefficients,
        #     indices,
        #     energy,
        #     error,
        #     gradient_norm,
        #     selected_gradients,
        #     # inv_hessian,
        #     gradients,
        #     nfevs,
        #     ngevs,
        #     nits,
        #     energy_opt_iters,
        #     shots_iters
        # )
        ansatz = None
        energy_change = 0
        sel_gradients = []
        initial_data = IterationData(
            ansatz,
            energy,
            error,
            energy_change,
            gradient_norm,
            sel_gradients,
            gradients,
            nfevs,
            ngevs,
            nits,
            
            energies_statevector,
            energies_uniform,
            energies_vmsa,
            energies_vpsr,
            std_uniform,
            std_vmsa,
            std_vpsr,
            shots_uniform,
            shots_vmsa,
            shots_vpsr
        )

        self.evolution.its_data.append(initial_data)

        return energy
    
    def close(self, success):
        self.result = self.evolution.last_it
        self.closed = True
        self.success = success
        # if file_name is not None:
        #     self.file_name = file_name


    def acc_depths(self, pool):
        """
        Outputs the list of accumulated depth through the iterations.
        Depth for iteration 0 (reference state), then for 1, then for 2, etc.
        Depth is the total number of gate layers - entangling or not, all gates are
        considered equal
        """
        assert pool.name == self.pool_name

        acc_depths = [0]
        ansatz_size = 0
        circuit = QuantumCircuit(pool.n)

        for iteration in self.evolution.its_data:
            indices = iteration.ansatz.indices
            coefficients = iteration.ansatz.coefficients
            new_indices = indices[ansatz_size:]
            new_coefficients = coefficients[ansatz_size:]
            ansatz_size += len(new_indices)

            new_circuit = pool.get_circuit(new_indices, new_coefficients)
            circuit = circuit.compose(new_circuit)
            depth = circuit.depth()
            acc_depths.append(depth)

        return acc_depths

    def acc_cnot_depths(self, pool, fake_params=False):
        """
        Outputs the list of accumulated CNOT depth through the iterations.
        Depth for iteration 0 (reference state), then for 1, then for 2, etc.
        All single qubit gates are ignored.
        """

        acc_depths = [0]
        ansatz_size = 0
        circuit = QuantumCircuit(pool.n)

        for iteration in self.evolution.its_data:
            indices = iteration.ansatz.indices
            coefficients = iteration.ansatz.coefficients
            new_indices = indices[ansatz_size:]
            new_coefficients = coefficients[ansatz_size:]
            ansatz_size += len(new_indices)

            if fake_params:
                # Sometimes if the coefficient is too small Openfermion will read the operator as zero, so this is
                # necessary for the circuit functions not to raise an error
                new_coefficients = [np.random.rand() for _ in coefficients]

            new_circuit = pool.get_circuit(new_indices, new_coefficients)
            circuit = circuit.compose(new_circuit)
            qasm_circuit = get_qasm(circuit)
            depth = cnot_depth(qasm_circuit, self.n)
            acc_depths.append(depth)

        return acc_depths

    def acc_cnot_counts(self, pool, fake_params=False):

        acc_counts = [0]
        ansatz_size = 0
        count = 0

        for iteration in self.evolution.its_data:
            indices = iteration.ansatz.indices
            coefficients = iteration.ansatz.coefficients
            new_indices = indices[ansatz_size:]
            new_coefficients = coefficients[ansatz_size:]
            ansatz_size += len(new_indices)

            if fake_params:
                # Sometimes if the coefficient is too small Openfermion will read the operator as zero, so this is
                # necessary for the circuit functions not to raise an error
                new_coefficients = [np.random.rand() for _ in coefficients]

            new_circuit = pool.get_circuit(new_indices, new_coefficients)
            qasm_circuit = get_qasm(new_circuit)
            count += cnot_count(qasm_circuit)
            acc_counts.append(count)

        return acc_counts
    
    @property
    def current(self):
        if self.evolution.its_data:
            return self.evolution.last_it
        else:
            return IterationData(energy=self.initial_energy)