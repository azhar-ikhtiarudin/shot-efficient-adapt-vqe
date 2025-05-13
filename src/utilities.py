
import re
import numpy as np
from openfermion import (
    count_qubits,
    FermionOperator,
    QubitOperator,
    get_fermion_operator,
    InteractionOperator,
    jordan_wigner,
    hermitian_conjugated,
    normal_ordered
)
from qiskit import QuantumCircuit
from openfermion.ops.representations import DiagonalCoulombHamiltonian, PolynomialTensor
from openfermion.ops.operators import FermionOperator, QubitOperator, BosonOperator, QuadOperator
import qiskit
from qiskit.qasm3 import dumps
from qiskit.quantum_info import Pauli, SparsePauliOp

I = SparsePauliOp("I")
X = SparsePauliOp("X")
Y = SparsePauliOp("Y")
Z = SparsePauliOp("Z")

pauliX = SparsePauliOp("X")
pauliY = SparsePauliOp("Y")
pauliZ = SparsePauliOp("Z")



def get_probability_distribution(counts, NUM_SHOTS, N):
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


def get_eigenvalues(pauli_strings):
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


def find_substrings(main_string, hamiltonian, checked=[]):
    """
    Finds and groups all the strings in a Hamiltonian that only differ from
    main_string by identity operators.

    Arguments:
      main_string (str): a Pauli string (e.g. "XZ)
      hamiltonian (dict): a Hamiltonian (with Pauli strings as keys and their
        coefficients as values)
      checked (list): a list of the strings in the Hamiltonian that have already
        been inserted in another group

    Returns:
      grouped_operators (dict): a dictionary whose keys are boolean strings
        representing substrings of the main_string (e.g. if main_string = "XZ",
        "IZ" would be represented as "01"). It includes all the strings in the
        hamiltonian that can be written in this form (because they only differ
        from main_string by identities), except for those that were in checked
        (because they are already part of another group of strings).
      checked (list):  the same list passed as an argument, with extra values
        (the strings that were grouped in this function call).
    """

    grouped_operators = {}

    # Go through the keys in the dictionary representing the Hamiltonian that
    # haven't been grouped yet, and find those that only differ from main_string
    # by identities
    for pauli_string in hamiltonian:

        if pauli_string not in checked:
            # The string hasn't been grouped yet

            if all(
                (op1 == op2 or op2 == "I")
                for op1, op2 in zip(main_string, pauli_string)
            ):
                # The string only differs from main_string by identities

                # Represent the string as a substring of the main one
                boolean_string = "".join(
                    [
                        str(int(op1 == op2))
                        for op1, op2 in zip(main_string, pauli_string)
                    ]
                )

                # Add the boolean string representing this string as a key to
                # the dictionary of grouped operators, and associate its
                # coefficient as its value
                grouped_operators[boolean_string] = hamiltonian[pauli_string]

                # Mark the string as grouped, so that it's not added to any
                # other group
                checked.append(pauli_string)

    return grouped_operators, checked

def get_qasm(qc):
    """
    Converts a Qiskit QuantumCircuit to qasm.
    Args:
        qc (QuantumCircuit): a Qiskit QuantumCircuit

    Returns:
        qasm (str): the QASM string for this circuit
    """

    if int(qiskit.__version__[0]) >= 1:
        qasm = dumps(qc)
    else:
        qasm = qc.qasm()

    return qasm


def bfgs_update(hk, gfkp1, gfk, xkp1, xk):
    """
    Performs a BFGS update.

    Arguments:
        hk (np.ndarray): the previous inverse Hessian (iteration k)
        gfkp1 (np.ndarray): the new gradient vector (iteration k + 1)
        gfk (np.ndarray): the old gradient vector (iteration k)
        xkp1 (np.ndarray): the new parameter vector (iteration k + 1)
        xk (np.ndarray):  the old parameter vector (iteration k)

    Returns:
        hkp1 (np.darray): the new inverse Hessian (iteration k + 1)
    """

    gfkp1 = np.array(gfkp1)
    gfk = np.array(gfk)
    xkp1 = np.array(xkp1)
    xk = np.array(xk)

    n = len(xk)
    id_mat = np.eye(n, dtype=int)

    sk = xkp1 - xk
    yk = gfkp1 - gfk

    rhok_inv = np.dot(yk, sk)
    if rhok_inv == 0.:
        rhok = 1000.0
        print("Divide-by-zero encountered: rhok assumed large")
    else:
        rhok = 1. / rhok_inv

    a1 = id_mat - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
    a2 = id_mat - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
    hkp1 = np.dot(a1, np.dot(hk, a2)) + (rhok * sk[:, np.newaxis] *
                                         sk[np.newaxis, :])

    return hkp1


def get_operator_qubits(operator):
    """
    Obtains the support of an operator.

    Args:
        operator (Union[FermionOperator, QubitOperator]): the operator in question

    Returns:
        qubits (Set): List containing the indices of the qubits in which operator acts on non-trivially
    """
    qubits = set()

    for string in list(operator.terms.keys()):
        for qubit, pauli in string:
            if qubit not in qubits:
                qubits.add(qubit)

    return qubits


def remove_z_string(operator):
    """
    Removes the anticommutation string from Jordan-Wigner transformed excitations. This is equivalent to removing
    all Z operators.
    This function does not change the original operator.

    Args:
        operator (Union[FermionOperator, QubitOperator]): the operator in question

    Returns:
        new_operator (Union[FermionOperator, QubitOperator]): the same operator, with Pauli-Zs removed
    """

    if isinstance(operator, QubitOperator):
        qubit_operator = operator
    else:
        qubit_operator = jordan_wigner(operator)

    new_operator = QubitOperator()

    for term in qubit_operator.get_operators():

        coefficient = list(term.terms.values())[0]
        pauli_string = list(term.terms.keys())[0]

        new_pauli = QubitOperator((), coefficient)

        for qubit, operator in pauli_string:
            if operator != 'Z':
                new_pauli *= QubitOperator((qubit, operator))

        new_operator += new_pauli

    return new_operator


def cnot_depth(qasm, n):
    """
    Counts the depth of a circuit on n qubits represented by a QASM string, considering only cx gates.
    Circuit must be decomposed into a cx + single qubit rotations gate set.

    Aguments:
        qasm (str): the QASM representation of the circuit
        n (int): the number of qubits
    Returns:
        The CNOT depth of the circuit
    """
    # n = int(re.search(r"(?<=q\[)[0-9]+(?=\])", qasm.splitlines()[2]).group())
    depths = [0 for _ in range(n)]

    for line in qasm.splitlines()[3:]:
        # Remove ;
        line = line[:-1]

        # Split line by spaces
        line_elems = line.split(" ")

        # First element is operation type
        op = line_elems[0]
        if op[:2] != "cx":
            continue

        # Next elements are qubits
        qubits = [
            int(re.search(r"[0-9]+", qubit_string).group())
            for qubit_string in line_elems[1:]
        ]

        max_depth = max([depths[qubit] for qubit in qubits])
        new_depth = max_depth + 1

        for qubit in qubits:
            depths[qubit] = new_depth

    return max(depths)


def cnot_count(qasm):
    """
    Counts the CNOTs in a circuit represented by a QASM string.
    """
    count = 0

    for line in qasm.splitlines()[3:]:
        # Remove ;
        line = line[:-1]
        line_elems = line.split(" ")
        op = line_elems[0]

        if op[:2] == "cx":
            count += 1

    return count




def qe_circuit(source_orbs, target_orbs, theta, n, big_endian=False):
    """
    Creates a qubit excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612

    Arguments:
        source_orbs (list): the spin-orbitals from which the excitation removes electrons
        target_orbs (list): the spin-orbitals to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    if len(source_orbs) == 2:
        return double_qe_circuit(source_orbs, target_orbs, theta, n, big_endian)
    else:
        return single_qe_circuit(source_orbs, target_orbs, theta, n, big_endian)


def double_qe_circuit(source_orbs, target_orbs, theta, n, big_endian=False):
    """
    Creates a qubit excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612

    Arguments:
        source_orbs (list): the spin-orbitals from which the excitation removes electrons
        target_orbs (list): the spin-orbitals to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    a, b = source_orbs
    c, d = target_orbs

    if big_endian:
        # Qiskit's default is little endian - switch
        a = n - a - 1
        b = n - b - 1
        c = n - c - 1
        d = n - d - 1

    qc = QuantumCircuit(n)

    qc.cx(a, b)
    qc.cx(c, d)
    qc.x(b)
    qc.x(d)
    qc.cx(a, c)
    qc.ry(2 * theta / 8, a)

    qc.h(b)
    qc.cx(a, b)
    qc.h(d)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, d)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.h(c)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, c)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, d)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.ry(-2 * theta / 8, a)

    qc.h(d)
    qc.h(b)
    qc.rz(+np.pi / 2, c)
    qc.cx(a, c)

    qc.rz(-np.pi / 2, a)
    qc.rz(+np.pi / 2, c)
    qc.ry(+np.pi / 2, c)

    qc.x(b)
    qc.x(d)
    qc.cx(a, b)
    qc.cx(c, d)

    return qc

def double_qe_circuit_edit(source_orbs, target_orbs, theta, n, big_endian=False):
    """
    Creates a qubit excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612

    Arguments:
        source_orbs (list): the spin-orbitals from which the excitation removes electrons
        target_orbs (list): the spin-orbitals to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    a, b = source_orbs
    c, d = target_orbs

    if big_endian:
        # Qiskit's default is little endian - switch
        a = n - a - 1
        b = n - b - 1
        c = n - c - 1
        d = n - d - 1

    qc = QuantumCircuit(n)

    qc.cx(a, b)
    qc.cx(c, d)
    #
    qc.x(b)
    qc.x(d)
    #
    qc.cx(a, c)
    #
    qc.ry(-theta/2, a)
    qc.h(b)
    #
    qc.cx(a, b)
    #
    qc.ry(theta / 4, a)
    qc.h(d)
    #
    qc.cx(a, d)
    #
    qc.ry(-theta / 4, a)
    #
    qc.cx(a, b)
    #
    qc.ry(theta / 4, a)
    qc.h(c)
    #
    qc.cx(a, c)
    qc.ry(-theta / 4, a)
    #
    qc.cx(a, b)
    qc.ry(theta / 4, a)
    #
    qc.cx(a, d) 
    #
    qc.ry(-theta / 4, a)
    qc.h(d) 
    # 
    qc.cx(a, b)
    #
    qc.ry(theta / 4, a)
    qc.h(b)
    qc.rz(-np.pi / 2, c)
    #
    qc.cx(a, c)
    #
    qc.rz(+np.pi / 2, a)
    qc.rz(-np.pi / 2, c)
    #
    qc.x(b)
    qc.ry(-np.pi / 2, c)
    qc.x(d)
    #
    qc.cx(a, b)
    qc.cx(c, d)

    return qc


def double_qe_circuit_backup(source_orbs, target_orbs, theta, n, big_endian=False):
    """
    Creates a qubit excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612

    Arguments:
        source_orbs (list): the spin-orbitals from which the excitation removes electrons
        target_orbs (list): the spin-orbitals to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    a, b = source_orbs
    c, d = target_orbs

    if big_endian:
        # Qiskit's default is little endian - switch
        a = n - a - 1
        b = n - b - 1
        c = n - c - 1
        d = n - d - 1

    qc = QuantumCircuit(n)

    qc.cx(a, b)
    qc.cx(c, d)
    qc.x(b)
    qc.x(d)
    qc.cx(a, c)
    qc.ry(-2 * theta / 8, a)

    qc.h(b)
    qc.cx(a, b)
    qc.h(d)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, d)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, b)
    qc.h(c)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, c)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, b)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, d)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, b)
    qc.ry(2 * theta / 8, a)

    qc.h(d)
    qc.h(b)
    qc.rz(+np.pi / 2, c)
    qc.cx(a, c)

    qc.rz(-np.pi / 2, a)
    qc.rz(+np.pi / 2, c)
    qc.ry(+np.pi / 2, c)

    qc.x(b)
    qc.x(d)
    qc.cx(a, b)
    qc.cx(c, d)

    return qc


def single_qe_circuit(source_orb, target_orb, theta, n, big_endian=False):
    """
    Creates a qubit excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612
    Example: if source_orb = [0] and target_orb = [1], this implements theta * 1/2 (X1 Y0 - Y1 X0)

    Arguments:
        source_orb (list): the spin-orbital from which the excitation removes electrons
        target_orb (list): the spin-orbital to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    a = source_orb[0]
    b = target_orb[0]

    if big_endian:
        a = n - a - 1
        b = n - b - 1

    qc = QuantumCircuit(n)

    qc.rz(np.pi / 2, a)
    qc.rx(np.pi / 2, a)
    qc.rx(np.pi / 2, b)
    qc.cx(a, b)

    qc.rx(theta, a)
    qc.rz(theta, b)
    qc.cx(a, b)

    qc.rx(-np.pi / 2, b)
    qc.rx(-np.pi / 2, a)
    qc.rz(-np.pi / 2, a)

    return qc



def normalize_op(operator):
    """
    Normalize Qubit or Fermion Operator by forcing the absolute values of the coefficients to sum to zero.
    This function modifies the operator.

    Arguments:
        operator (Union[FermionOperator,QubitOperator]): the operator to normalize

    Returns:
        operator (Union[FermionOperator,QubitOperator]): the same operator, now normalized0
    """

    if operator:
        coeff = 0
        for t in operator.terms:
            coeff_t = operator.terms[t]
            # coeff += np.abs(coeff_t * coeff_t)
            coeff += np.abs(coeff_t)

        # operator = operator/np.sqrt(coeff)
        operator = operator / coeff

    return operator

def get_hf_det(electron_number, qubit_number):
    """
    Get the Hartree Fock ket |1>|1>...|0>|0>.

    Arguments:
    electron_number (int): the number of electrons of the molecule.
    qubit_number (int): the number of qubits necessary to represent the molecule
      (equal to the number of spin orbitals we're considering active).

    Returns:
    reference_ket (list): a list of lenght qubit_number, representing the
      ket of the adequate computational basis state in big-endian ordering.
    """

    # Consider occupied the lower energy orbitals, until enough one particle
    # states are filled
    reference_ket = [1 for _ in range(electron_number)]

    # Consider the remaining orbitals empty
    reference_ket += [0 for _ in range(qubit_number - electron_number)]

    return reference_ket



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


def to_qiskit_pauli(letter):
    """
    Transforms a letter representing a Pauli operator to the corresponding
    Qiskit observable.

    Arguments:
        letter (str): the letter representing the Pauli operator
    Returns:
        qiskit_Pauli (PauliOp): the corresponding operator in Qiskit
    """
    if letter == "X":
        qiskit_pauli = X
    elif letter == "Y":
        qiskit_pauli = Y
    elif letter == "Z":
        qiskit_pauli = Z
    else:
        raise ValueError(
            "Letter isn't recognized as a Pauli operator" " (must be X, Y or Z)."
        )

    return qiskit_pauli


def to_qiskit_term(of_term, n, switch_endianness):
    """
    Transforms an Openfermion term into a Qiskit Operator.
    Only works for individual Pauli strings. For generic operators, see to_qiskit_operator.

    Arguments:
        of_term (QubitOperator): a Pauli string multiplied by a coefficient, given as an Openfermion operator
        n (int): the size of the qubit register
        switch_endianness (bool): whether to revert the endianness
    Returns:
        qiskit_op (PauliSumOp): the original operator, represented in Qiskit
    """

    pauli_strings = list(of_term.terms.keys())

    if len(pauli_strings) > 1:
        raise ValueError(
            "Input must consist of a single Pauli string."
            " Use to_qiskit_operator for other operators."
        )
    pauli_string = pauli_strings[0]

    coefficient = of_term.terms[pauli_string]
    

    

    qiskit_op = None

    previous_index = -1
    # print("pauli string", pauli_string)
    for qubit_index, pauli in pauli_string:
        # print("AA")
        id_count = qubit_index - previous_index - 1
    
        if switch_endianness:
            new_ops = to_qiskit_pauli(pauli)
            for _ in range(id_count):
                new_ops = new_ops ^ I
            if qiskit_op is None:
                qiskit_op = new_ops
            else:
                qiskit_op = new_ops ^ qiskit_op
        else:
            new_ops = (I ^ id_count) ^ to_qiskit_pauli(pauli)
            qiskit_op = qiskit_op ^ new_ops
        # print("--qiskit_op-1", qiskit_op)
        previous_index = qubit_index

    id_count = (n - previous_index - 1)
    # print("BB", switch_endianness)
    if switch_endianness:
        # print(id_count)
        for _ in range(id_count):
            # print("--I", I)
            # print("--qiskit_op-2", qiskit_op)
            qiskit_op = I ^ qiskit_op
    else:
        for _ in range(id_count):
            qiskit_op = qiskit_op ^ I
    
    # print("coefficient", coefficient)
    # print("qiskit_op", qiskit_op)

    qiskit_op = coefficient * qiskit_op

    return qiskit_op


def to_qiskit_operator(of_operator, n=None, little_endian=True):
    """
    Transforms an Openfermion operator into a Qiskit Operator.

    Arguments:
        of_operator (QubitOperator): a linear combination of Pauli strings as an Openfermion operator
        n (int): the size of the qubit register
        little_endian (bool): whether to revert use little endian ordering
    Returns:
        qiskit_operator (PauliSumOp): the original operator, represented in Qiskit
    """

    # If of_operator is an InteractionOperator, shape it into a FermionOperator
    if isinstance(of_operator, InteractionOperator):
        of_operator = get_fermion_operator(of_operator)

    # print(of_operator)
    if not n:
        n = count_qubits(of_operator)

    # print("N qubits: ",n)
    # Now use the Jordan Wigner transformation to map the FermionOperator into
    # a QubitOperator
    if isinstance(of_operator, FermionOperator):
        of_operator = jordan_wigner(of_operator)

    qiskit_operator = None

    # Iterate through the terms in the operator. Each is a Pauli string
    # multiplied by a coefficient
    for term in of_operator.get_operators():
        # print("==TERM==",term)
        if list(term.terms.keys())==[()]:
            # coefficient = term.terms[term.terms.keys()[0]]
            coefficient = term.terms[list(term.terms.keys())[0]]

            result = I
            # print("n", n)
            for _ in range(n-1):
                result = result ^ I

            qiskit_term = coefficient * result
            # print("empty qiskit term", qiskit_term)
        else:
            qiskit_term = to_qiskit_term(term, n, little_endian)
            # print("non empty qiskit term",qiskit_term)

        if qiskit_operator is None:
            qiskit_operator = qiskit_term
        else:
            qiskit_operator += qiskit_term

    return qiskit_operator


def group_hamiltonian(hamiltonian):
    """
    Organizes a Hamiltonian into groups where strings only differ from
    identities, so that the expectation values of all the strings in each
    group can be calculated from the same measurement array.

    Arguments:
      hamiltonian (dict): a dictionary representing a Hamiltonian, with Pauli
        strings as keys and their coefficients as values.

    Returns:
      grouped_hamiltonian (dict): a dictionary of subhamiltonians, each of
        which includes Pauli strings that only differ from each other by
        identities.
        The keys of grouped_hamiltonian are the main strings of each group: the
        ones with least identity terms. The value associated to a main string is
        a dictionary, whose keys are boolean strings representing substrings of
        the respective main string (with 1 where the Pauli is the same, and 0
        where it's identity instead). The values are their coefficients.
    """
    grouped_hamiltonian = {}
    checked = []

    # Go through the hamiltonian, starting by the terms that have less
    # identity operators
    for main_string in sorted(
        hamiltonian, key=lambda pauli_string: pauli_string.count("I")
    ):

        # Call find_substrings to find all the strings in the dictionary that
        # only differ from main_string by identities, and organize them as a
        # dictionary (grouped_operators)
        grouped_operators, checked = find_substrings(main_string, hamiltonian, checked)

        # Use the dictionary as a value for the main_string key in the
        # grouped_hamiltonian dictionary
        grouped_hamiltonian[main_string] = grouped_operators

        # If all the strings have been grouped, exit the for cycle
        if len(checked) == len(hamiltonian.keys()):
            break

    return grouped_hamiltonian


def convert_hamiltonian(openfermion_hamiltonian):
    """
    Formats a qubit Hamiltonian obtained from openfermion, so that it's a suitable
    argument for functions such as measure_expectation_estimation.

    Arguments:
      openfermion_hamiltonian (openfermion.qubitOperator): the Hamiltonian.

    Returns:
      formatted_hamiltonian (dict): the Hamiltonian as a dictionary with Pauli
        strings (eg 'YXZI') as keys and their coefficients as values.
    """

    formatted_hamiltonian = {}
    qubit_number = count_qubits(openfermion_hamiltonian)

    # Iterate through the terms in the Hamiltonian
    for term in openfermion_hamiltonian.get_operators():

        operators = []
        coefficient = list(term.terms.values())[0]
        pauli_string = list(term.terms.keys())[0]
        previous_qubit = -1

        for qubit, operator in pauli_string:

            # If there are qubits in which no operations are performed, add identities
            # as necessary, to make sure that the length of the string will match the
            # number of qubits
            identities = qubit - previous_qubit - 1
            if identities > 0:
                operators.append("I" * identities)

            operators.append(operator)
            previous_qubit = qubit

        # Add final identity operators if the string still doesn't have the
        # correct length (because no operations are performed in the last qubits)
        operators.append("I" * (qubit_number - previous_qubit - 1))

        formatted_hamiltonian["".join(operators)] = coefficient

    return formatted_hamiltonian

def string_to_matrix(pauli_string,little_endian=False):
    """
    Converts a Pauli string to its matrix form.

    Arguments:
        pauli_string (str): the Pauli string (e.g. "IXYIZ")
        little_endian (bool): whether the input ket is in little endian notation
    Returns:
        matrix (np.ndarray): the corresponding matrix, in the computational basis
    """

    print("-- String to Matrix Function --")
    print("pauli string: ", pauli_string)
    if little_endian:
        pauli_string = pauli_string[::-1]

    matrix = np.array([1])
    print("Initial Matrix", matrix)

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
        print("----matrix pauli iter:", matrix)

    return matrix





def hamiltonian_to_matrix(hamiltonian):
    """
    Convert a Hamiltonian (from OpenFermion) to matrix form.

    Arguments:
      hamiltonian (openfermion.InteractionOperator): the Hamiltonian to be
        transformed.

    Returns:
      matrix (np.ndarray): the Hamiltonian, as a matrix in the computational
        basis

    """

    qubit_number = hamiltonian.n_qubits
    # print("Second Quantization Hamiltonian:\n", hamiltonian)

    hamiltonian = jordan_wigner(hamiltonian)
    print("\nQubit Hamiltonian:\n", hamiltonian)

    formatted_hamiltonian = convert_hamiltonian(hamiltonian)
    print("\nFormatted Hamiltonian:\n", formatted_hamiltonian)

    grouped_hamiltonian = group_hamiltonian(formatted_hamiltonian)
    print("\nGrouped Hamiltonian:\n", grouped_hamiltonian)

    matrix = np.zeros((2**qubit_number, 2**qubit_number), dtype=complex)
    print("\nMatrix Size:", matrix.shape)
    # print("Matrix:", matrix)

    # Iterate through the strings in the Hamiltonian, adding the respective
    # contribution to the matrix
    for string in grouped_hamiltonian:
        print("\n-String:", string)

        for substring in grouped_hamiltonian[string]:
            print("--Substring:", substring)
            # pauli = "".join(
            #     "I" * (not int(b)) + a * int(b) for (a, b) in zip(string, substring)
            # )
            
            pauli = ""
            for a, b in zip(string, substring):
                print("a, b:", a, b)
                if int(b) == 0:
                    pauli += "I"
                else:
                    pauli += a

            print("--Pauli", pauli)
            
            matrix_from_pauli = string_to_matrix(pauli) * grouped_hamiltonian[string][substring]
            print("\nMatrix from Pauli", matrix_from_pauli)

            matrix += matrix_from_pauli

    return matrix



def create_qes(p, q, r, s):
    """
    Creates all unique qubit excitations acting on the set of spin-orbitals p,q,r,s.

    If aaaa or bbbb, all possible source/orbital pair combinations are valid.
    In this case, all the ifs apply and we get 6 distinct operators.

    In the other cases, only two source/orbital pair combinations are valid.
    In this case, only one of the ifs applies and we get 2 distinct operators.

    Arguments:
        p, q, r, s (int): the spin-orbital indices

    Returns:
        q_operators (list): list of lists containing pairs of qubit excitations. If p,q,r,s are aaaa or bbbb, the list
            contains three pairs of qubit excitations. Otherwise it contains one.
        orbs (list): list of lists containing pairs of source/target orbitals. Length: same as described above.
            The source (target) orbitals for q_operators[0] are returned in orbs[0][0] (orbs[1][0]).
            The source (target) orbitals for q_operators[1] are returned in orbs[0][1] (orbs[1][1]).
    """


    q_operators = []
    orbs = []
    if (p + r) % 2 == 0:
        # pqrs is abab or baba, or aaaa or bbbb

        f_operator_1 = FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)))
        # f_operator_2 = FermionOperator(((p, 0), (q, 1), (r, 1), (s, 0)))
        f_operator_2 = FermionOperator(((q, 1), (r, 1), (p, 0), (s, 0)))

        f_operator_1 -= hermitian_conjugated(f_operator_1)
        f_operator_2 -= hermitian_conjugated(f_operator_2)

        f_operator_1 = normal_ordered(f_operator_1)
        f_operator_2 = normal_ordered(f_operator_2)

        q_operator_1 = remove_z_string(f_operator_1)
        q_operator_2 = remove_z_string(f_operator_2)

        source_orbs = [[r, s], [p, s]]
        target_orbs = [[p, q], [q, r]]

        q_operators.append([q_operator_1, q_operator_2])
        orbs.append([source_orbs, target_orbs])

    if (p + q) % 2 == 0:
        # aabb or bbaa, or aaaa or bbbb

        # f_operator_1 = FermionOperator(((p, 1), (q, 0), (r, 1), (s, 0)))
        f_operator_1 = FermionOperator(((p, 1), (r, 1), (q, 0), (s, 0)))
        # f_operator_2 = FermionOperator(((p, 0), (q, 1), (r, 1), (s, 0)))
        f_operator_2 = FermionOperator(((q, 1), (r, 1), (p, 0), (s, 0)))

        f_operator_1 -= hermitian_conjugated(f_operator_1)
        f_operator_2 -= hermitian_conjugated(f_operator_2)

        f_operator_1 = normal_ordered(f_operator_1)
        f_operator_2 = normal_ordered(f_operator_2)

        q_operator_1 = remove_z_string(f_operator_1)
        q_operator_2 = remove_z_string(f_operator_2)

        source_orbs = [[q, s], [p, s]]
        target_orbs = [[p, r], [q, r]]

        q_operators.append([q_operator_1, q_operator_2])
        orbs.append([source_orbs, target_orbs])

    if (p + s) % 2 == 0:
        # abba or baab, or aaaa or bbbb

        f_operator_1 = FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)))
        # f_operator_2 = FermionOperator(((p, 1), (q, 0), (r, 1), (s, 0)))
        f_operator_2 = FermionOperator(((p, 1), (r, 1), (q, 0), (s, 0)))

        f_operator_1 -= hermitian_conjugated(f_operator_1)
        f_operator_2 -= hermitian_conjugated(f_operator_2)

        f_operator_1 = normal_ordered(f_operator_1)
        f_operator_2 = normal_ordered(f_operator_2)

        q_operator_1 = remove_z_string(f_operator_1)
        q_operator_2 = remove_z_string(f_operator_2)

        source_orbs = [[r, s], [q, s]]
        target_orbs = [[p, q], [p, r]]

        q_operators.append([q_operator_1, q_operator_2])
        orbs.append([source_orbs, target_orbs])

    return q_operators, orbs