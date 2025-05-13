from src.pools import QE
from src.molecules import create_h2, create_h3, create_lih
from src.hamiltonian import h_lih
from src.utilities import to_qiskit_operator
from algorithms.adapt_vqe import AdaptVQE

from openfermion.transforms import jordan_wigner
from openfermion.utils import commutator

from qiskit.quantum_info import SparsePauliOp, PauliList
import numpy as np


if __name__ == '__main__':    
    r = 0.742
    molecule = create_h2(r)

    # Hamiltonian
    fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
    qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)


    # Pool
    pool = QE(molecule=molecule)
    gradient_list = []
    pauli_list = PauliList(["IIII"])
    coeff_list = np.array([])


    # OBTAIN GRADIENT OBSERVABLE AND GROUPING
    for i in range(len(pool.operators)):
        print(f"\nPool-{i}")
        gradient = commutator(qubit_hamiltonian, pool.operators[i].q_operator)
        gradient_qiskit = to_qiskit_operator(gradient)
        gradient_list.append(gradient_qiskit)
        
        pauli_list = pauli_list.insert(len(pauli_list), gradient_qiskit._pauli_list)
        coeff_list = np.concatenate((coeff_list, gradient_qiskit.coeffs))

    pauli_list = pauli_list.delete(0)

    
    # GRADIENT OBSERVABLE 
    gradient_obs_list = SparsePauliOp(pauli_list, coeff_list)
    commuted_gradient_obs_list = gradient_obs_list.group_commuting(qubit_wise=True)


    # QUANTUM MEASUREMENT
    gradient_value = {}
    coeff_value = {}

    for clique_idx in range(len(commuted_gradient_obs_list)):

        # print(f'\nClique-{clique_idx}')

        clique_measure = 2

        for commuted_term_idx in range(len(commuted_gradient_obs_list[clique_idx])):

            commuted_pauli = commuted_gradient_obs_list[clique_idx][commuted_term_idx] 
            pauli_string = commuted_gradient_obs_list[clique_idx][commuted_term_idx].paulis[0].to_label()
            pauli_coeffs = commuted_gradient_obs_list[clique_idx][commuted_term_idx].coeffs[0].real
            
            gradient_value[pauli_string] = clique_measure
            coeff_value[pauli_string] = pauli_coeffs

    
    print(gradient_value)
    print(coeff_value)
    
    # CALCULATE GRADIENT FROM MEASUREMENT RESULTS

    gradient_result_list = []
    for gradient in gradient_list:
        
        print(gradient.paulis)

        gradient_result = 0
        for pauli in gradient.paulis:
            gradient_result += gradient_value[str(pauli)] * coeff_value[str(pauli)]
        
        gradient_result_list.append(gradient_result)
    
    print("\nFinal Results:")
    print(gradient_result_list)

    qiskit_hamiltonian = to_qiskit_operator(qubit_hamiltonian)
    commuted_qiskit_hamiltonian = qiskit_hamiltonian.group_commuting(qubit_wise=True)
    print(commuted_qiskit_hamiltonian)
