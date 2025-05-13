# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:27:05 2022

@author: mafal
"""

import numpy as np
import scipy

from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix

# Pauli operators (two-dimensional) as matrices
pauliX = np.array([[0, 1],
                   [1, 0]],
                  dtype=complex)
pauliZ = np.array([[1, 0],
                   [0, -1]],
                  dtype=complex)
pauliY = np.array([[0, -1j],
                   [1j, 0]],
                  dtype=complex)


def index_to_ket(index, dimension, little_endian=False):
    """
    Transforms a computational basis statevector described by the index of its
    nonzero entry into the corresponding ket.

    Arguments:
        index (int): the index of the non-zero element of the computational
          basis state.
        dimension (int): the dimension of the Hilbert space
        little_endian (bool): whether to output the ket using little endian notation

    Returns:
        ket (list): the corresponding ket as a list of length dimension

    """

    vector = [0 for _ in range(index)] + [1] + [1 for _ in range(dimension - index - 1)]
    ket = vector_to_ket(vector)

    if little_endian:
        ket = ket[::-1]

    return ket


def vector_to_ket(state_vector, little_endian=False):
    """
    Transforms a vector representing a basis state to the corresponding ket.

    Arguments:
        state_vector (np.ndarray): computational basis vector in the 2^n dimensional
          Hilbert space
        little_endian (bool): whether to output the ket using little endian notation
    Returns:
        ket (list): a list of length n representing the corresponding ket
    """

    dim = len(state_vector)
    ket = []

    while dim > 1:
        if any(state_vector[i] for i in range(int(dim / 2))):
            # Ket is of the form |0>|...>.

            # Fix |0> as the msq.
            ket.append(0)

            # Get the vector representing the state of the remaining qubits.
            state_vector = state_vector[:int(dim / 2)]

        else:
            # Ket is of the form |1>|...>.

            # Fix |0> as the msq.
            ket.append(1)

            # Get the vector representing the state of the remaining qubits.
            state_vector = state_vector[int(dim // 2):]

        dim = dim / 2

    if little_endian:
        ket = ket[::-1]

    return ket


def string_to_index(string, little_endian=False):
    """
    Turns a string representing a computational basis state into the index of the
    non-null element of the corresponding statevector
    e.g. "010" -> 2

    Arguments:
        string (str): a computational basis state
        little_endian (bool): whether the input ket is in little endian notation
    Returns:
        (int) The index of the position of "1" in the statevector representing this state in the Z basis
    """

    if little_endian:
        string = string[::-1]

    n = len(string)
    dim = 2 ** n

    index = 0
    # Go through qubits, left to right. 
    for state in string:
        # Reduce dimension to stop including this qubit
        dim = dim / 2

    if state == "1":
        # State is |1>=[0,1]. Put the index in the subspace corresponding to the 
        # correct state of this particular qubit (second half)
        # If state is 0, index doesn't change (first half)
        index += dim

    return int(index)


def string_to_matrix(pauli_string,little_endian=False):
    """
    Converts a Pauli string to its matrix form.

    Arguments:
        pauli_string (str): the Pauli string (e.g. "IXYIZ")
        little_endian (bool): whether the input ket is in little endian notation
    Returns:
        matrix (np.ndarray): the corresponding matrix, in the computational basis
    """

    if little_endian:
        pauli_string = pauli_string[::-1]

    matrix = np.array([1])

    # Iteratively construct the matrix, going through each single qubit Pauli term
    for pauli in pauli_string:
        if pauli == "I":
            matrix = np.kron(matrix, np.identity(2))
        elif pauli == "X":
            matrix = np.kron(matrix, pauliX)
        elif pauli == "Y":
            matrix = np.kron(matrix, pauliY)
        elif pauli == "Z":
            matrix = np.kron(matrix, pauliZ)

    return matrix


def ket_to_vector(ket,little_endian=False):
    """
    Transforms a ket representing a basis state to the corresponding state vector.

    Arguments:
        ket (list): a list of length n representing the ket
        little_endian (bool): whether the input ket is in little endian notation

    Returns:
        state_vector (np.ndarray): the corresponding basis vector in the
            2^n dimensional Hilbert space
    """

    if little_endian:
        ket = ket[::-1]

    state_vector = [1]

    # Iterate through the ket, calculating the tensor product of the qubit states
    for i in ket:
        qubit_vector = [not i, i]
        state_vector = np.kron(state_vector, qubit_vector)

    return state_vector


def calculate_overlap(state1, state2):
    """
    Calculates the overlap between two states, given their coordinates.

    Arguments:
        state1 (np.ndarray): the coordinates of one of the states in some
          orthonormal basis
        state2 (np.ndarray): the coordinates of the other state, in the same
          basis

    Returns:
        overlap (float): the overlap between two states (absolute value of the
        inner product).
    """

    bra = np.conj(state1)
    ket = state2
    overlap = np.abs(np.dot(bra, ket))

    return overlap


def state_energy(state, hamiltonian):
    """
    Calculates the exact energy in a specific state.

    Arguments:
        state (np.ndarray): the state in which to obtain the expectation value.
        hamiltonian (dict): the Hamiltonian of the system.

    Returns:
        exact_energy (float): the energy expectation value in the state.
    """

    exact_energy = 0

    # Obtain the theoretical expectation value for each Pauli string in the
    # Hamiltonian by matrix multiplication, and perform the necessary weighed
    # sum to obtain the energy expectation value.
    for pauli_string in hamiltonian:
        ket = np.array(state, dtype=complex)
        bra = np.conj(ket)

        ket = np.matmul(string_to_matrix(pauli_string), ket)
        expectation_value = np.real(np.dot(bra, ket))

        exact_energy += \
            hamiltonian[pauli_string] * expectation_value

    return exact_energy


def create_unitary(coefficients,
                   operators,
                   n):
    """
    Create a unitary e^(C_N*Op_N)...e^(C_1*Op_1).

    Arguments:
        coefficients (list): the coefficients of the exponentials
        operators (list): the matrices representing the operators
        n (int): the dimension
    """

    unitary = scipy.sparse.identity(n)

    # Apply e ** (coefficient * operator) for each operator in
    # operators, following the order of the list
    for coefficient, operator in zip(coefficients, operators):
        # Multiply the operator by the respective coefficient
        operator = coefficient * operator

        unitary = expm_multiply(operator, unitary)

    return unitary


def revert_endianness(statevector):
    """
    Reverts a statevector assuming the endianness is switched: qubit k is now qubit n-1-k, where n is the total number
    of qubits of the system.

    Arguments:
        statevector (csc_matrix): The statevector to be reverted.
    Returns:
        csc_matrix: The reverted statevector.
    """

    new_statevector = csc_matrix((2 ** n, 1), dtype=complex)

    for i, amp in enumerate(statevector):
        # Write computational basis state corresponding to this entry as string
        cb_state = bin(int(i))[2:]
        cb_state = "0" * (n - len(cb_state)) + cb_state

        # Revert endianness
        new_cb_state = cb_state[::-1]

        # Convert back to decimal
        new_index = int(new_cb_state, 2)

        # Obtain corresponding computational basis state
        ket = index_to_ket(i, n)

        # Fill the entry with the value
        new_statevector[new_index] = amp[0, 0]

    return new_statevector