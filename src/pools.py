
import abc
import numpy as np
import itertools

from copy import copy
from openfermion import FermionOperator, QubitOperator, hermitian_conjugated, normal_ordered, jordan_wigner, get_sparse_operator
from openfermion.transforms import freeze_orbitals

from .utilities import get_operator_qubits, remove_z_string, cnot_depth, cnot_count, qe_circuit, normalize_op, create_qes
from .circuits import pauli_exp_circuit, ovp_ceo_circuit, mvp_ceo_circuit

from scipy.sparse import csc_matrix, issparse, identity
from scipy.sparse.linalg import expm, expm_multiply

from qiskit import QuantumCircuit

class OpType:
    FERMIONIC = 0
    QUBIT = 1


class ImplementationType:
    SPARSE = 0
    QISKIT = 1


class PoolOperator(metaclass=abc.ABCMeta):

    def __init__(self, operator, n, tag, frozen_orbitals=[], cnots=None, cnot_depth=None, parents=None,
                 source_orbs=None, target_orbs=None, ceo_type=None):
        """
        Arguments:
            operator(Union[FermionOperator,QubitOperator]: the operator we want to represent
            n (int): number of qubits the operator acts on. Note that this includes identity operators - it's dependent
            on system size, not operator support
            tag (int): number identifying position in pool
            frozen_orbitals (list): indices of orbitals that are considered to be permanently occupied. Note that
                virtual orbitals are not yet implemented.
            cnots (int): number of CNOTs in the circuit implementation of this operator
            cnot_depth (int): CNOT depth in the circuit implementation of this operator
            parents (list): indices of operators that this one derives from (in the case of CEOs, where operator is
                linear combination of parents)
            source_orbs (list): spin-orbitals from which the operator removes fermions
            target_orbs (list): spin-orbitals to which the operator adds fermions
            ceo_type (str): "sum" or "diff", defining the type of OVP-CEO when applicable

        Note: Operator may be modified by class methods!
        If this were not desired, we could do self._f_operator = operator * 1. This creates a new copy of the operator.
        """

        if isinstance(operator, FermionOperator):
            if frozen_orbitals:
                self._f_operator = freeze_orbitals(operator, frozen_orbitals)
            else:
                self._f_operator = operator

            self._q_operator = None
            self.op_type = OpType.FERMIONIC

        elif isinstance(operator, QubitOperator):
            self._f_operator = None
            self._q_operator = operator
            self.op_type = OpType.QUBIT
            self.cnots = cnots
            self.cnot_depth = cnot_depth
            self.parents = parents

        else:
            raise TypeError("Expected Fermion or QubitOperator, not {}."
                            .format(type(operator).__name__))

        self.qubits = get_operator_qubits(operator)
        self.n = n
        self.tag = tag
        self.coef = None
        self.imp_operator = None  # implemented version (e.g. Qiskit Operator)
        self.exp_operator = None  # exponential version (e.g. trotter circuit)
        self.grad_meas = None  # gradient observable
        self.twin_string_ops = []  # operators in the same pool with the exact same Pauli strings
        self.source_orbs = source_orbs
        self.target_orbs = target_orbs
        self.ceo_type = ceo_type

    def __str__(self):

        return self.operator.__str__()

    def __eq__(self, other):

        if isinstance(other, PoolOperator):
            return (self.operator == other.operator or
                    self.operator == - other.operator)

        return False

    def arrange(self):
        """
        Arrange self.
        If self is a fermionic operator $\tau$, it will be made into a proper
        anti-hermitian pool operator $\tau$ - hc($\tau$) and normal-ordered.
        Both fermionic and qubit operators are normalized also.

        Return value: True if the operator is nontrivial, true if it's trivial

        This does not change the state.
        """

        if self.op_type == OpType.FERMIONIC:
            # Subtract hermitian conjugate to make the operator anti Hermitian
            self._f_operator -= hermitian_conjugated(self._f_operator)

            # Normal order the resulting operator so that we have a consistent ordering
            self._f_operator = normal_ordered(self._f_operator)

        if not self.operator.many_body_order():
            # Operator acts on 0 qubits; discard
            return False

        self.normalize()

        return True

    def normalize(self):
        """
        Normalize self, so that the sum of the absolute values of coefficients is one.
        """

        self._f_operator = normalize_op(self._f_operator)
        self._q_operator = normalize_op(self._q_operator)

    def create_qubit(self):
        """
        Create a qubit version of the fermion operator.
        """

        if not self._q_operator:
            self._q_operator = normalize_op(jordan_wigner(self._f_operator))

    def create_sparse(self):
        """
        Obtain sparse matrix representing the space, in the proper dimension (might be higher than the effective
        dimension of operator)
        """
        self.imp_operator = get_sparse_operator(self.q_operator, self.n)

    @property
    def f_operator(self):
        return self._f_operator

    @property
    def q_operator(self):
        if not self._q_operator:
            self.create_qubit()
        return self._q_operator

    @property
    def operator(self):
        if self.op_type == OpType.QUBIT:
            return self._q_operator
        if self.op_type == OpType.FERMIONIC:
            return self._f_operator



class OperatorPool(metaclass=abc.ABCMeta):
    name = None

    def __init__(self, molecule=None, frozen_orbitals=[], n=None, source_ops=None):
        """
        Arguments:
            molecule (PyscfMolecularData): the molecule for which we will use the pool
            frozen_orbitals (list): indices of orbitals that are considered to be permanently occupied. Note that
                virtual orbitals are not yet implemented.
            n (int): number of qubits the operator acts on. Note that this includes identity operators - it's dependent
            on system size, not operator support
            source_ops (list): the operators to generate the pool from, if tiling.
        """

        if self.name is None:
            raise NotImplementedError("Subclasses must define a pool name.")

        if self.op_type == OpType.QUBIT:
            self.has_qubit = True
        else:
            self.has_qubit = False

        self.frozen_orbitals = frozen_orbitals
        self.molecule = molecule
        self.source_ops = source_ops

        if molecule is None:
            assert n is not None
            self.n = n
        else:
            self.n_so = molecule.n_orbitals  # Number of spatial orbitals
            self.n = molecule.n_qubits - len(frozen_orbitals)  # Number of qubits = 2*n_so

        self.operators = []
        self._ops_on_qubits = {}

        self.create_operators()
        self.eig_decomp = [None for _ in range(self.size)]
        self.squared_ops = [None for _ in range(self.size)]

        # Get range of parent doubles - i.e., double excitations from which pool is generated by taking sums/differences
        # Only applicable if we have CEO pool.
        for i in range(self.size):
            if len(self.get_qubits(i)) == 4:
                break
        if self.name[:3] in ["DVG", "DVE"]:
            self.parent_range = range(i, self.parent_pool.size)
        else:
            self.parent_range = []

    def __add__(self, other):

        assert isinstance(other, OperatorPool)
        assert self.n == other.n
        assert self.op_type == other.op_type

        pool = copy(self)
        pool.operators = copy(self.operators)

        for operator in other.operators:
            if operator not in pool.operators:
                pool.operators.append(operator)

        pool.name = pool.name + "_+_" + other.name
        pool.eig_decomp = pool.eig_decomp + other.eig_decomp
        pool.couple_exchanges = self.couple_exchanges or other.couple_exchanges

        return pool

    def __str__(self):

        if self.op_type == OpType.QUBIT:
            type_str = "Qubit"
        if self.op_type == OpType.FERMIONIC:
            type_str = "Fermionic"

        text = f"{type_str} pool with {self.size} operators\n"

        for i, operator in enumerate(self.operators):
            text += f"{i}:\n{str(operator)}\n\n"

        return text

    def add_operator(self, new_operator, cnots=None, cnot_depth=None, parents=None, source_orbs=None, target_orbs=None,
                     ceo_type=None):
        """
        Arguments:
            new_operator (Union[PoolOperator,FermionOperator,QubitOperator]): operator to add to pool
            cnots (int): number of CNOTs in the circuit implementation of this operator
            cnot_depth (int): CNOT depth in the circuit implementation of this operator
            parents (list): indices of operators that this one derives from (in the case of CEOs, where operator is a
                linear combination of parents)
            source_orbs (list): spin-orbitals from which the operator removes fermions
            target_orbs (list): spin-orbitals to which the operator adds fermions
            ceo_type (str): "sum" or "diff", defining the type of OVP-CEO when applicable
        """

        if not isinstance(new_operator, PoolOperator):
            new_operator = PoolOperator(new_operator,
                                        self.n,
                                        self.size,
                                        self.frozen_orbitals,
                                        cnots,
                                        cnot_depth,
                                        parents,
                                        source_orbs,
                                        target_orbs,
                                        ceo_type)

        is_nontrivial = new_operator.arrange()

        if is_nontrivial and new_operator not in self.operators:
            self.operators.append(new_operator)
            position = len(self.operators) - 1
            return position

        return None

    @property
    def imp_type(self):
        return self._imp_type

    @imp_type.setter
    def imp_type(self, imp_type):
        if imp_type not in [ImplementationType.SPARSE]:
            raise ValueError("Argument isn't a valid implementation type.")

        self._imp_type = imp_type

    @abc.abstractmethod
    def create_operators(self):
        """
        Fill self.operators list with PoolOperator objects
        """
        pass

    @abc.abstractmethod
    def get_circuit(self, coefficients, indices):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments, as a Qiskit QuantumCircuit.
        Arguments:
            indices (list)
            coefficients (list)
        """
        pass

    def create_sparse(self):

        for operator in self.operators:
            operator.create_sparse()

    def create_eig_decomp(self, index):
        """
        Create eigendecomposition for operator represented by the given index (which identifies its place in the pool).
        Having the eigendecomposition facilitates implementing the exponential of the operator, because you can
        simply do a basis rotation, exponentiate a diagonal matrix, and revert the basis rotation.
        The exponential of a diagonal matrix is easy to obtain because you simply exponentiate the diagonal entries.
        Once you have the eigendecomposition, the calculations become much faster, because you do two matrix
        multiplications instead of one matrix exponentiation (which is significantly more complex).
        However, it might take quite some time to create the eigendecomposition for a complete pool. This becomes
        intractable for 14 qubits or more.
        """

        if self.eig_decomp[index] is None:
            print("Diagonalizing operator...")
            dense_op = self.get_imp_op(index).todense()
            # eigh is for Hermitian matrices, H is skew-Hermitian. Multiply -1j, undo later
            hermitian_op = -1j * dense_op
            w, v = np.linalg.eigh(hermitian_op)  # hermitian_op = v * diag(w) * inv(v)
            v[abs(v) < 1e-16] = 0
            v = csc_matrix(v)
            self.eig_decomp[index] = 1j * w, v

    def create_eig_decomps(self):
        """
        Create eigendecomposition for operator represented by the given index (which identifies its place in the pool).
        Having the eigendecomposition facilitates implementing the exponential of the operator, because you can
        simply do a basis rotation, exponentiate a diagonal matrix, and revert the basis rotation.
        The exponential of a diagonal matrix is easy to obtain because you simply exponentiate the diagonal entries.
        Once you have the eigendecomposition, the calculations become much faster, because you do two matrix
        multiplications instead of one matrix exponentiation (which is significantly more complex).
        However, it might take quite some time to create the eigendecomposition for a complete pool. This becomes
        intractable for 14 qubits or more.
        """

        for index in range(self.size):
            self.create_eig_decomp(index)

    def get_op(self, index):
        """
        Returns the operator specified by its index in the pool.
        """

        if self.op_type == OpType.FERMIONIC:
            return self.get_f_op(index)
        else:
            return self.get_q_op(index)

    def get_qubits(self, index):
        """
        Returns list of qubits in which the operator specified by this index acts non trivially.
        """
        return self.operators[index].qubits

    def get_parents(self, index):
        """
        Applicable only to CEO operators.
        Returns the QEs from which the operator derives (by taking linear combination).
        """
        return self.operators[index].parents

    def get_ops_on_qubits(self, qubits):
        """
        Returns the indices of the operators in the pool that act on the given qubits.
        """

        # Use this instead of directly accessing self._ops_on_qubits - the key must be sorted, you know you'll forget
        if not self._ops_on_qubits:
            raise ValueError("Operators have not been associated to qubits in this pool.")
        return self._ops_on_qubits[str(sorted(qubits))]

    def get_twin_ops(self, index):
        """
        Returns the indices of the operators in the pool that act on the same qubits as the operator identified by index
        """
        return self.operators[index].twin_string_ops

    def get_imp_op(self, index):
        """
        Returns implemented version of operator (depends on implementation type).
        """

        if self.operators[index].imp_operator is None:
            # print("ImplementationType.SPARSE", ImplementationType.SPARSE)
            # print("self.imp_type", self.imp_type)
            if self.imp_type == ImplementationType.SPARSE:
                self.operators[index].create_sparse()
            else:
                raise AttributeError("PoolOperator does not have imp_operator attribute because an implementation type "
                                     "hasn't been set for this pool. "
                                     "Please choose an implementation by setting pool.imp_type.")

        return self.operators[index].imp_operator

    def get_f_op(self, index):
        """
        Get fermionic operator labeled by index.
        """
        return self.operators[index].f_operator

    def get_q_op(self, index):
        """
        Get qubit operator labeled by index.
        """
        return self.operators[index].q_operator

    def get_exp_op(self, index, coefficient=None):
        """
        Get exponential of operator labeled by index.
        """
        if self.op_type == ImplementationType.SPARSE:
            return expm(coefficient * self.operators[index].imp_operator)
        else:
            raise ValueError

    def square(self, index):
        """
        Get square of operator labeled by index.
        It can be useful to store the value to make the computation faster.
        """

        op = self.get_imp_op(index)
        self.squared_ops[index] = op.dot(op)

        return self.squared_ops[index]

    def expm(self, coefficient, index):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient.
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.

        Arguments:
            coefficient (float)
            index (int)
        """
        assert self.op_type == ImplementationType.SPARSE

        if self.eig_decomp[index] is None:
            return expm(coefficient * self.operators[index].imp_operator)
        else:
            diag, unitary = self.eig_decomp[index]
            exp_diag = np.exp(coefficient * diag)
            exp_diag = exp_diag.reshape(exp_diag.shape[0], 1)
            return unitary.dot(np.multiply(exp_diag, unitary.T.conjugate().todense()))

    def expm_mult(self, coefficient, index, other):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient, multiplying
        another pool operator (indexed "other").
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.

        Arguments:
            coefficient (float)
            index (int)
            other (csc_matrix)
        """

        assert self.imp_type == ImplementationType.SPARSE



        if self.eig_decomp[index] is None:
            print(issparse(other))
            if not issparse(other):
                other = csc_matrix(other)
            
            print("Operator index :", self.operators[index])
            print("Operator IMP :", self.operators[index].imp_operator)
            multiply_operator = get_sparse_operator(self.operators[index].q_operator, self.n)
            print("Operator:", multiply_operator)
            
            # breakpoint()

            return expm_multiply(coefficient * multiply_operator, other)
            # return expm_multiply(coefficient * self.operators[index].imp_operator, other)
        
        else:
            if issparse(other):
                other = other.todense()
            diag, unitary = self.eig_decomp[index]
            exp_diag = np.exp(coefficient * diag)
            exp_diag = exp_diag.reshape(exp_diag.shape[0], 1)
            m = unitary.T.conjugate().dot(other)
            m = np.multiply(exp_diag, m)
            m = unitary.dot(m)
            m = m.real
            return m




    # def expm_matrix(self, coefficient, index):
    #     """
    #     Calculates the exponential of the operator defined by index, when multiplied by the coefficient.
    #     If an eigendecomposition of the operator exists, it will be used for increased efficiency.

    #     Arguments:
    #         coefficient (float)
    #         index (int)
    #     """
    #     # assert self.op_type == ImplementationType.SPARSE

    #     if self.eig_decomp[index] is None:
    #         return expm(coefficient * self.operators[index].create_sparse())
    #     else:
    #         diag, unitary = self.eig_decomp[index]
    #         exp_diag = np.exp(coefficient * diag)
    #         exp_diag = exp_diag.reshape(exp_diag.shape[0], 1)
    #         return unitary.dot(np.multiply(exp_diag, unitary.T.conjugate().todense()))

    # def expm_mult_matrix(self, coefficient, index, other):
    #     """
    #     Calculates the exponential of the operator defined by index, when multiplied by the coefficient, multiplying
    #     another pool operator (indexed "other").
    #     If an eigendecomposition of the operator exists, it will be used for increased efficiency.

    #     Arguments:
    #         coefficient (float)
    #         index (int)
    #         other (csc_matrix)
    #     """

    #     # assert self.imp_type == ImplementationType.SPARSE

    #     if self.eig_decomp[index] is None:
    #         if not issparse(other):
    #             other = csc_matrix(other)
    #         return expm_multiply(coefficient * self.operators[index].create_sparse(), other)
    #     else:
    #         if issparse(other):
    #             other = other.todense()
    #         diag, unitary = self.eig_decomp[index]
    #         exp_diag = np.exp(coefficient * diag)
    #         exp_diag = exp_diag.reshape(exp_diag.shape[0], 1)
    #         m = unitary.T.conjugate().dot(other)
    #         m = np.multiply(exp_diag, m)
    #         m = unitary.dot(m)
    #         m = m.real
    #         return m

    def get_cnots(self, index):
        """
        Obtain number of CNOTs required in the circuit implementation of the operator labeled by index.
        If index is a list, it must represent a MVP-CEO.
        """

        if isinstance(index,list):

            # Make sure all operators are qubit excitations acting on the same qubits. If they are, the number of CNOTs
            #required in the circuit implementation is the same regardless of the number of operators
            op_qubits = [self.get_qubits(i) for i in index]
            assert all(qubits == op_qubits[0] for qubits in op_qubits)
            assert all([i in self.parent_range for i in index])
            index = index[0]

        return self.operators[index].cnots

    def get_cnot_depth(self, index):
        """
        Obtain CNOT depth of the circuit implementation of the operator labeled by index.
        """
        return self.operators[index].cnot_depth

    def get_grad_meas(self, index):
        """
        Obtain observable corresponding to the (energy) gradient of the operator labeled by index.
        """
        return self.operators[index].grad_meas

    def store_grad_meas(self, index, measurement):
        """
        Set the observable corresponding to the (energy) gradient of the operator labeled by index.
        """
        self.operators[index].grad_meas = measurement

    @abc.abstractproperty
    def op_type(self):
        """
        Type of pool (qubit/fermionic).
        """
        pass

    @property
    def size(self):
        """
        Number of operators in pool.
        """
        return len(self.operators)

    @property
    def exp_operators(self):
        """
        List of pool operators, in their exponential versions.
        """
        return [self.get_exp_op(i) for i in range(self.size)]

    @property
    def imp_operators(self):
        """
        List of pool operators, in their implemented versions.
        """
        return [self.get_imp_op(i) for i in range(self.size)]

    def cnot_depth(self, coefficients, indices):
        """
        Obtain CNOT depth of the circuit implementation of the ansatz represented by input lists of coefficients
        and pool operator indices.
        """
        circuit = self.get_circuit(coefficients, indices)
        return cnot_depth(circuit.qasm())

    def depth(self, coefficients, indices):
        """
        Obtain depth of the circuit implementation of the ansatz represented by input lists of coefficients
        and pool operator indices.
        """
        circuit = self.get_circuit(coefficients, indices)
        return circuit.depth

    def cnot_count(self, coefficients, indices):
        """
        Obtain CNOT count of the circuit implementation of the ansatz represented by input lists of coefficients
        and pool operator indices.
        """
        circuit = self.get_circuit(coefficients, indices)
        return cnot_count(circuit.qasm())


class SD(OperatorPool):
    """Basic Single and Double Excitations operator. Occupied to virtual only"""
    name = 'sd'

    def create_operators(self):
        self.create_singles()
        self.create_doubles()
    
    def create_singles(self):
        """Create one-body SD operators"""
        for p in range(0, self.molecule.n_electrons):
            for q in range(self.molecule.n_electrons, self.n):
                if (p+q) % 2 == 0:
                    self.add_operator(FermionOperator(((p, 1), (q, 0))))
    
    def create_doubles(self):
        """Create two-body SD operators"""
        for p in range(0, self.molecule.n_electrons):
            for q in range(p+1, self.molecule.n_electrons):
                for r in range(self.molecule.n_electrons, self.n-1):
                    for s in range(r+1, self.n):
                        if (p + q + r + s) % 2 != 0:
                            continue
                        if p%2 + q%2 != r%2 + s%2:
                            continue
                        self.add_operator(FermionOperator(((p, 1), (q, 1), (r, 0), (s,0))))
    
    @property
    def op_type(self):
        return OpType.FERMIONIC
    
    def get_circuit(self):
        pass


class GSD(OperatorPool):
    """
    GSD where repeated indices (e.g. 0^ 3^ 0 5 ) are not allowed
    """
    name = 'gsd'

    def create_operators(self):
        self.create_singles()
        self.create_doubles()

    def create_singles(self):
        """Create one-body GSD operators"""
        for p in range(0, self.n):
            for q in range(p+1,self.n):
                if (p+q)%2 == 0:
                    self.add_operator(FermionOperator(((p, 1), (1,0))), source_orbs=[q], target_orbs=[p])

    def create_doubles(self):
        """Create two-body GSD operators"""
        pq = -1
        for p in range(0, self.n_so):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_so):
                qa = 2 * q
                qb = 2 * q + 1

                pq += 1

                rs = -1
                for r in range(0, self.n_so):
                    ra = 2 * r
                    rb = 2 * r + 1

                    for s in range(r, self.n_so):
                        sa = 2 * s
                        sb = 2 * s + 1

                        rs += 1

                        if pq > rs:
                            continue

                        term_1 = FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)),
                                                2 / np.sqrt(12))
                        term_1 += FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)),
                                                2 / np.sqrt(12))
                        term_1 += FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)),
                                                1 / np.sqrt(12))
                        term_1 += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)),
                                                1 / np.sqrt(12))
                        term_1 += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)),
                                                1 / np.sqrt(12))
                        term_1 += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)),
                                                1 / np.sqrt(12))

                        term_2 = FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)),
                                                1 / 2.0)
                        term_2 += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)),
                                                1 / 2.0)
                        term_2 += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)),
                                                -1 / 2.0)
                        term_2 += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)),
                                                -1 / 2.0)

                        for term in [term_1, term_2]:
                            self.add_operator(term)

    @property
    def op_type(self):
        return OpType.FERMIONIC
    
    def get_circuit(self, indices, coefficients):
        raise NotImplementedError

class GSD1(OperatorPool):
    name = 'gsd'
    """
    GSD where repeated indices (e.g. 0^ 3^ 0 5 ) are allowed
    """
    def create_operators(self):
        """Create pool operators"""
        self.create_singles()
        self.create_doubles()

    def create_singles(self):
        """
        Create one-body GSD operators.
        """

        for p in range(0, self.n_so):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_so):
                qa = 2 * q
                qb = 2 * q + 1

                terms = []

                terms.append(FermionOperator(((pa, 1), (qa, 0))))
                terms.append(FermionOperator(((pb, 1), (qb, 0))))

                for term in terms:
                    self.add_operator(term)

    def create_doubles(self):
        """
        Create two-body GSD operators.
        """
        pq = -1
        for p in range(0, self.n_so):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_so):
                qa = 2 * q
                qb = 2 * q + 1

                pq += 1

                rs = -1
                for r in range(0, self.n_so):
                    ra = 2 * r
                    rb = 2 * r + 1

                    for s in range(r, self.n_so):
                        sa = 2 * s
                        sb = 2 * s + 1

                        rs += 1

                        if pq > rs:
                            continue

                        terms = []

                        terms.append(FermionOperator
                                    (((ra, 1), (pa, 0), (sa, 1), (qa, 0))))
                        terms.append(FermionOperator
                                    (((rb, 1), (pb, 0), (sb, 1), (qb, 0))))

                        terms.append(FermionOperator
                                    (((ra, 1), (pa, 0), (sb, 1), (qb, 0))))
                        terms.append(FermionOperator
                                    (((ra, 1), (pb, 0), (sb, 1), (qa, 0))))

                        terms.append(FermionOperator
                                    (((rb, 1), (pb, 0), (sa, 1), (qa, 0))))
                        terms.append(FermionOperator
                                    (((rb, 1), (pa, 0), (sa, 1), (qb, 0))))

                        for term in terms:
                            self.add_operator(term)

    @property
    def op_type(self):
        return OpType.FERMIONIC
    
    def get_circuit(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments, as a Qiskit QuantumCircuit.
        The function is specific for pools where the generators are sums of commuting Paulis, such as GSD or Pauli pools
        Arguments:
            indices (list)
            coefficients (list)
        """

        circuit = QuantumCircuit(self.n)

        for i, (index, coefficient) in enumerate(zip(indices, coefficients)):
            qubit_operator = coefficient * self.operators[index].q_operator
            qc = pauli_exp_circuit(qubit_operator, self.n, revert_endianness=True)

            circuit = circuit.compose(qc)
            circuit.barrier()

        return circuit



class SingletGSD(OperatorPool):

    name = 'singlet_gsd'

    def create_operators(self):
        self.create_singles()
        self.create_doubles()
    
    def create_singles(self):
        """Create one-body singlet GSD operators"""
        for p in range(0, self.n_so):
            pa = 2*p
            pb = 2*p + 1
            
            for q in range(p, self.n_so):
                qa = 2*q
                qb = 2*q + 1

                term = FermionOperator(((pa, 1), (qa,0)))
                term += FermionOperator(((pb, 1), (qb,0)))

                self.add_operator(term)
    

    def create_doubles(self):
        """
        Create two-body singlet GSD operators.
        """
        pq = -1
        for p in range(0, self.n_so):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_so):
                qa = 2 * q
                qb = 2 * q + 1

                pq += 1

                rs = -1
                for r in range(0, self.n_so):
                    ra = 2 * r
                    rb = 2 * r + 1

                    for s in range(r, self.n_so):
                        sa = 2 * s
                        sb = 2 * s + 1

                        rs += 1

                        if pq > rs:
                            continue

                        term_1 = FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)),
                                                2 / np.sqrt(12))
                        term_1 += FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)),
                                                2 / np.sqrt(12))
                        term_1 += FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)),
                                                1 / np.sqrt(12))
                        term_1 += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)),
                                                1 / np.sqrt(12))
                        term_1 += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)),
                                                1 / np.sqrt(12))
                        term_1 += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)),
                                                1 / np.sqrt(12))

                        term_2 = FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)),
                                                1 / 2.0)
                        term_2 += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)),
                                                1 / 2.0)
                        term_2 += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)),
                                                -1 / 2.0)
                        term_2 += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)),
                                                -1 / 2.0)

                        for term in [term_1, term_2]:
                            self.add_operator(term)

    @property
    def op_type(self):
        return OpType.FERMIONIC

    def get_circuit(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments, as a Qiskit QuantumCircuit.
        The function is specific for pools where the generators are sums of commuting Paulis, such as GSD or Pauli pools
        Arguments:
            indices (list)
            coefficients (list)
        """

        raise NotImplementedError


class SpinCompGSD(OperatorPool):
    """
    Spin-complemented generalized single and double excitations.
    """
    name = "sc_gsd"

    def create_operators(self):
        """
        Create pool operators and insert them into self.operators (list).
        """

        self.create_singles()
        self.create_doubles()

    def create_singles(self):
        """
        Create one-body spin-complemented GSD operators.
        """

        for p in range(0, self.n_so):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_so):
                qa = 2 * q
                qb = 2 * q + 1

                term = FermionOperator(((pa, 1), (qa, 0)))
                term += FermionOperator(((pb, 1), (qb, 0)))

                self.add_operator(term)

    def create_doubles(self):
        """
        Create two-body spin-complemented GSD operators.
        """

        pq = -1
        for p in range(0, self.n_so):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_so):
                qa = 2 * q
                qb = 2 * q + 1

                pq += 1

                rs = -1
                for r in range(0, self.n_so):
                    ra = 2 * r
                    rb = 2 * r + 1

                    for s in range(r, self.n_so):
                        sa = 2 * s
                        sb = 2 * s + 1

                        rs += 1

                        if pq > rs:
                            continue

                        term_1 = FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)))
                        term_1 += FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)))

                        term_2 = FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)))
                        term_2 += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)))

                        term_3 = FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)))
                        term_3 += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)))

                        for term in [term_1, term_2, term_3]:
                            self.add_operator(term)

    @property
    def op_type(self):
        return OpType.FERMIONIC

    def get_circuit(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments, as a Qiskit QuantumCircuit.
        The function is specific for pools where the generators are sums of commuting Paulis, such as GSD or Pauli pools
        Arguments:
            indices (list)
            coefficients (list)
        """

        raise NotImplementedError


class PauliPool(SingletGSD):

    name = 'pauli_pool'

    def create_operators(self):
        super().create_operators()

        pool_operators = self.operators

        self.operators = []

        for pool_operator in pool_operators:
            
            fermionic_op = pool_operator.operator
            qubit_op = jordan_wigner(fermionic_op)

            for pauli in qubit_op.terms:
                qubit_op = QubitOperator(pauli, 1j)
                new_operator = PoolOperator(qubit_op, self.n, self.size)
                self.add_operator(new_operator)

    def expm(self, coefficient, index):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient.
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.
        Otherwise, a trigonometric formula leveraging the structure of the operators is used. This is quite faster
            than using generic matrix exponentiation methods.

        Arguments:
            coefficient (float)
            index (int)
        """
        if self.eig_decomp[index] is not None:
            return super().expm(coefficient, index)
        op = self.get_imp_op(index)
        n, n = op.shape
        exp_op = np.cos(coefficient) * identity(n) + np.sin(coefficient) * op
        return exp_op

    def expm_mult(self, coefficient, index, other):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient, multiplying
        another pool operator (indexed "other").
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.
        Otherwise, a trigonometric formula leveraging the structure of the operators is used. This is quite faster
        than using generic matrix exponentiation methods.

        Arguments:
            coefficient (float)
            index (int)
            other (csc_matrix)
        """
        if self.eig_decomp[index] is not None:
            return super().expm_mult(coefficient, index, other)
        '''
        exp_op = self.expm(coefficient, index)
        m = exp_op.dot(other)
        '''
        # It's faster to do product first, then sums; this way we never do
        # matrix-matrix operations, just matrix-vector
        op = self.get_imp_op(index)
        m = op.dot(other)
        m = np.cos(coefficient) * other + np.sin(coefficient) * m
        # '''
        return m

    def get_circuit(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments.
        Function for pools where the generators are sums of commuting Paulis.
        E.g. GSD, Pauli pool
        """
        circuit = QuantumCircuit(self.n)

        for i, (index, coefficient) in enumerate(zip(indices, coefficients)):
            qubit_operator = coefficient * self.operators[index].q_operator
            qc = pauli_exp_circuit(qubit_operator, self.n, revert_endianness=True)

            circuit = circuit.compose(qc)
            circuit.barrier()

        return circuit
    
    def get_parameterized_circuit(self, indices, coefficients, parameters):
        circuit = QuantumCircuit(self.n)

        for i, (index, coefficient) in enumerate(zip(indices, coefficients)):
            qubit_operator = coefficient * self.operators[index].q_operator
            qc = pauli_exp_circuit(qubit_operator, self.n, revert_endianness=True)

            circuit = circuit.compose(qc)
            circuit.barrier()

        return circuit


    @property
    def op_type(self):
        return OpType.QUBIT





class NoZPauliPool1(PauliPool):
    """
    Same as PauliPool, but with Z operators removed (no Jordan-Wigner anticommutation string).
    It derives from SingletGSD, which is nice to make sure we are doing it right. However, it is inefficient to
    construct it this way. Check NoZPauliPool (below) for a better implementation
    """

    name = "no_z_pauli_pool"

    def create_operators(self):
        """
        Create pool operators and insert them into self.operators (list).
        """

        # Create Pauli pool (with Z)
        super().create_operators()

        # Go through the pool operators
        # We might me removing operators from self.operators so we're creating a copy
        pool_operators = self.operators
        self.operators = []

        for pool_operator in pool_operators:
            qubit_op = pool_operator.operator
            qubit_op = list(qubit_op.get_operators())[0]
            pauli_string = list(qubit_op.terms.keys())[0]

            new_qubit_op = QubitOperator((), 1j)
            for qubit, operator in pauli_string:
                if operator != 'Z':
                    new_qubit_op *= QubitOperator((qubit, operator))

            new_operator = PoolOperator(new_qubit_op, self.n, self.size)
            self.add_operator(new_operator)


class NoZPauliPool(PauliPool):
    """
    Same as PauliPool, but with Z operators removed (no Jordan-Wigner anticommutation string).
    It is equivalent to NoZPauliPool1, but more efficient.
    """

    name = "no_z_pauli_pool"

    def create_operators(self):

        self.create_singles()
        self.create_doubles()

    def create_singles(self):

        for a, b in itertools.combinations(range(self.n), 2):
            parity = (a + b) % 2

            if parity == 0:
                # alpha -> alpha or beta -> beta.
                # Only these strings can possibly appear in GSD operators which preserve Sz spin
                self.add_operator(QubitOperator(((a, 'Y'), (b, 'X')), 1j), cnots=2, cnot_depth=2)
                self.add_operator(QubitOperator(((a, 'X'), (b, 'Y')), 1j), cnots=2, cnot_depth=2)

    def create_doubles(self):

        for a, b, c, d in itertools.combinations(range(self.n), 4):
            parity_sum = (a % 2 + b % 2 + c % 2 + d % 2)

            if parity_sum % 2 == 0:
                # Either all spin-orbitals are of the same type (all alpha / all beta), or they come in pairs (two
                # alpha, two beta).
                # Only these strings can possibly appear in GSD operators which preserve Sz spin.

                self.add_operator(QubitOperator(((a, 'Y'), (b, 'X'), (c, 'X'), (d, 'X')), 1j), cnots=6, cnot_depth=6)
                self.add_operator(QubitOperator(((a, 'X'), (b, 'Y'), (c, 'X'), (d, 'X')), 1j), cnots=6, cnot_depth=6)
                self.add_operator(QubitOperator(((a, 'X'), (b, 'X'), (c, 'Y'), (d, 'X')), 1j), cnots=6, cnot_depth=6)
                self.add_operator(QubitOperator(((a, 'X'), (b, 'X'), (c, 'X'), (d, 'Y')), 1j), cnots=6, cnot_depth=6)

                self.add_operator(QubitOperator(((a, 'X'), (b, 'Y'), (c, 'Y'), (d, 'Y')), 1j), cnots=6, cnot_depth=6)
                self.add_operator(QubitOperator(((a, 'Y'), (b, 'X'), (c, 'Y'), (d, 'Y')), 1j), cnots=6, cnot_depth=6)
                self.add_operator(QubitOperator(((a, 'Y'), (b, 'Y'), (c, 'X'), (d, 'Y')), 1j), cnots=6, cnot_depth=6)
                self.add_operator(QubitOperator(((a, 'Y'), (b, 'Y'), (c, 'Y'), (d, 'X')), 1j), cnots=6, cnot_depth=6)

    @property
    def op_type(self):
        return OpType.QUBIT

class QE1(GSD):
    """
    Pool consisting of qubit excitations, which are obtained by removing the Z strings from fermionic generalized
    single and double excitations. This class is pedagogical because it derives from GSD, but QE (below) is more
    efficient as it avoids implementing the GSD to begin with.
    """

    name = "QE"

    def create_operators(self):
        """
        Create pool operators and insert them into self.operators (list).
        """

        # Create gsd pool
        super().create_operators()

        # Store the gsd operators temporarily
        pool_operators = self.operators
        # print(pool_operators)

        # Empty operator list - we will fill it with qubit operators now
        self.operators = []

        for pool_operator in pool_operators:

            q_op = pool_operator.q_operator
            print("\nq_op:", q_op)

            new_operator = QubitOperator()

            for term in q_op.get_operators():
                print("term", term)

                coefficient = list(term.terms.values())[0]
                pauli_string = list(term.terms.keys())[0]

                new_pauli = QubitOperator((), coefficient)


                # Remove all Z strings
                for qubit, operator in pauli_string:
                    if operator != 'Z':
                        new_pauli *= QubitOperator((qubit, operator))
                        print("new pauli:", new_pauli)

                # print("new operator:", new_operator)
                new_operator += new_pauli

            new_operator = PoolOperator(new_operator, self.n, self.size)
            print("new operator:", new_operator)

            self.add_operator(new_operator)

    @property
    def op_type(self):
        return OpType.QUBIT

    def get_circuit(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments, as a Qiskit QuantumCircuit.
        The function is specific for pools where the generators are sums of commuting Paulis, such as GSD or Pauli pools
        Arguments:
            indices (list)
            coefficients (list)
        """

        raise NotImplementedError



class QE(OperatorPool):
    """
    Pool consisting of qubit excitations, which are obtained by removing the Z strings from fermionic generalized
    single and double excitations. Instead of building a GSD pool first, we create the operators by iterating through
    combinations of indices we know are associated with valid excitations. This is more efficient than QE1.
    """

    name = "QE"

    def __init__(self,
                 molecule=None,
                 couple_exchanges=False,
                 frozen_orbitals=[],
                 n=None,
                 source_ops=None):
        """
        Arguments:
            molecule (PyscfMolecularData): the molecule for which we will use the pool
            couple_exchanges (bool): whether to add all qubit excitations with nonzero gradient acting on the same
                qubits when a given double qubit excitation is added to the ansatz. If this flag is set to True,
                the pool will correspond to the MVP-CEO pool when used in ADAPT-VQE.
            frozen_orbitals (list): indices of orbitals that are considered to be permanently occupied. Note that
                virtual orbitals are not yet implemented.
            n (int): number of qubits the operator acts on. Note that this includes identity operators - it's dependent
            on system size, not operator support
            source_ops (list): the operators to generate the pool from, if tiling.
        """

        self.couple_exchanges = couple_exchanges

        if couple_exchanges:
            self.name = "MVP_CEO"

        super().__init__(molecule, frozen_orbitals, n=n, source_ops=source_ops)

    def create_operators(self):
        """
        Create pool operators and insert them into self.operators (list).
        """

        self.create_singles()
        self.create_doubles()

    def create_singles(self):
        """
        Create one-body qubit excitations.
        """

        for p in range(0, self.n):

            for q in range(p + 1, self.n):

                if (p + q) % 2 == 0:
                    f_operator = FermionOperator(((p, 1), (q, 0)))
                    f_operator -= hermitian_conjugated(f_operator)
                    f_operator = normal_ordered(f_operator)

                    q_operator = remove_z_string(f_operator)
                    pos = self.add_operator(q_operator, cnots=2, cnot_depth=2,
                                            source_orbs=[q], target_orbs=[p])
                    self._ops_on_qubits[str([p, q])] = [pos]

    def create_doubles(self):
        """
        Create two-body qubit excitations.
        """

        for p in range(0, self.n):

            for q in range(p + 1, self.n):

                for r in range(q + 1, self.n):

                    for s in range(r + 1, self.n):

                        if (p + q + r + s) % 2 != 0:
                            continue

                        # If aaaa or bbbb, all three of the following ifs apply, but there are only 3 distinct operators
                        # In the other cases, only one of the ifs applies, 2 distinct operators

                        new_positions = []
                        if (p + r) % 2 == 0:
                            # pqrs is abab or baba, or aaaa or bbbb

                            f_operator_1 = FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)))
                            f_operator_2 = FermionOperator(((q, 1), (r, 1), (p, 0), (s, 0)))

                            f_operator_1 -= hermitian_conjugated(f_operator_1)
                            f_operator_2 -= hermitian_conjugated(f_operator_2)

                            f_operator_1 = normal_ordered(f_operator_1)
                            f_operator_2 = normal_ordered(f_operator_2)

                            q_operator_1 = remove_z_string(f_operator_1)
                            q_operator_2 = remove_z_string(f_operator_2)

                            pos1 = self.add_operator(q_operator_1, cnots=13, cnot_depth=11,
                                                     source_orbs=[r, s], target_orbs=[p, q])

                            pos2 = self.add_operator(q_operator_2, cnots=13, cnot_depth=11,
                                                     source_orbs=[p, s], target_orbs=[q, r])

                            new_positions += [pos1, pos2]

                        if (p + q) % 2 == 0:
                            # aabb or bbaa, or aaaa or bbbb

                            f_operator_1 = FermionOperator(((p, 1), (r, 1), (q, 0), (s, 0)))
                            f_operator_2 = FermionOperator(((q, 1), (r, 1), (p, 0), (s, 0)))

                            f_operator_1 -= hermitian_conjugated(f_operator_1)
                            f_operator_2 -= hermitian_conjugated(f_operator_2)

                            f_operator_1 = normal_ordered(f_operator_1)
                            f_operator_2 = normal_ordered(f_operator_2)

                            q_operator_1 = remove_z_string(f_operator_1)
                            q_operator_2 = remove_z_string(f_operator_2)

                            pos1 = self.add_operator(q_operator_1, cnots=13, cnot_depth=11,
                                                     source_orbs=[q, s], target_orbs=[p, r])

                            pos2 = self.add_operator(q_operator_2, cnots=13, cnot_depth=11,
                                                     source_orbs=[p, s], target_orbs=[q, r])

                            new_positions += [pos1, pos2]

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

                            pos1 = self.add_operator(q_operator_1, cnots=13, cnot_depth=11,
                                                     source_orbs=[r, s], target_orbs=[p, q])

                            pos2 = self.add_operator(q_operator_2, cnots=13, cnot_depth=11,
                                                     source_orbs=[q, s], target_orbs=[p, r])

                            new_positions += [pos1, pos2]

                        new_positions = [pos for pos in new_positions if pos is not None]
                        self._ops_on_qubits[str([p, q, r, s])] = new_positions
                        if self.couple_exchanges:
                            for pos1, pos2 in itertools.combinations(new_positions, 2):
                                self.operators[pos1].twin_string_ops.append(pos2)
                                self.operators[pos2].twin_string_ops.append(pos1)

    @property
    def op_type(self):
        return OpType.QUBIT

    def expm(self, coefficient, index):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient.
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.
        Otherwise, a trigonometric formula leveraging the structure of the operators is used. This is quite faster
            than using generic matrix exponentiation methods.

        Arguments:
            coefficient (float)
            index (int)
        """
        if self.eig_decomp[index] is not None:
            return super().expm(index, coefficient)
        op = self.get_imp_op(index)
        n, n = op.shape
        exp_op = identity(n) + np.sin(coefficient) * op + (1 - np.cos(coefficient)) * self.square(index)
        return exp_op

    def expm_mult(self, coefficient, index, other):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient, multiplying
        another pool operator (indexed "other").
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.
        Otherwise, a trigonometric formula leveraging the structure of the operators is used. This is quite faster
        than using generic matrix exponentiation methods.

        Arguments:
            coefficient (float)
            index (int)
            other (csc_matrix)
        """
        if self.eig_decomp[index] is not None:
            return super().expm_mult(coefficient, index, other)
        '''
        exp_op = self.expm(index,coefficient)
        m = exp_op.dot(other)
        '''
        # It's faster to do product first, then sums; this way we never do matrix-matrix operations, just matrix-vector
        op = self.get_imp_op(index)
        m = op.dot(other)
        # In the following we can use self.square(index).dot(ket) instead of op.dot(m). But that's actually slightly
        # slower even if self.square(index) was already stored and we don't have to calculate it
        m = other + np.sin(coefficient) * m + (1 - np.cos(coefficient)) * op.dot(m)

        return m

    # def get_circuit(self, indices, coefficients, parameters):
    #     """
    #     Returns the circuit corresponding to the ansatz defined by the arguments.
    #     Function for the QE pool only.
    #     """

    #     circuit = QuantumCircuit(self.n)

    #     for i, (index, coefficient) in enumerate(zip(indices, coefficients)):
    #         operator = self.operators[index]
    #         source_orbs = operator.source_orbs
    #         target_orbs = operator.target_orbs
    #         qc = qe_circuit(source_orbs, target_orbs, parameters[i], self.n, big_endian=False)

    #         circuit = circuit.compose(qc)
    #         circuit.barrier()

    #     return circuit
    
    def get_circuit(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments.
        Function for the QE pool only.
        """

        circuit = QuantumCircuit(self.n)

        for i, (index, coefficient) in enumerate(zip(indices, coefficients)):
            operator = self.operators[index]
            source_orbs = operator.source_orbs
            target_orbs = operator.target_orbs
            qc = qe_circuit(source_orbs, target_orbs, coefficient, self.n, big_endian=True)

            circuit = circuit.compose(qc)
            circuit.barrier()

        return circuit


    def get_parameterized_circuit(self, indices, coefficients, parameters):

        circuit = QuantumCircuit(self.n)

        for i, (index, coefficients) in enumerate(zip(indices, coefficients)):
            operator = self.operators[index]
            source_orbs = operator.source_orbs
            target_orbs = operator.target_orbs
            qc = qe_circuit(source_orbs, target_orbs, parameters[i], self.n, big_endian=False)
            circuit = circuit.compose(qc)
            circuit.barrier()
        
        return circuit

    def get_circuit_unparameterized(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments.
        Function for the QE pool only.
        """

        circuit = QuantumCircuit(self.n)

        for i, (index, coefficient) in enumerate(zip(indices, coefficients)):
            operator = self.operators[index]
            source_orbs = operator.source_orbs
            target_orbs = operator.target_orbs
            qc = qe_circuit(source_orbs, target_orbs, coefficient, self.n, big_endian=True)

            circuit = circuit.compose(qc)
            circuit.barrier()

        return circuit



class QE_All(QE):
    """
    Same as QE, but here we consider excitations between any 2 orbital pairs (not Sz preserving necessarily).
    """

    name = "QE_ALL"

    def create_singles(self):
        """
        Create one-body qubit excitations
        """

        for p in range(0, self.n):

            for q in range(p + 1, self.n):
                f_operator = FermionOperator(((p, 1), (q, 0)))
                f_operator -= hermitian_conjugated(f_operator)
                f_operator = normal_ordered(f_operator)

                q_operator = remove_z_string(f_operator)
                pos = self.add_operator(q_operator, cnots=2, cnot_depth=2,
                                        source_orbs=[q], target_orbs=[p])
                self._ops_on_qubits[str([p, q])] = [pos]

    def create_doubles(self):
        """
        Create two-body qubit excitations
        """

        for p in range(0, self.n):

            for q in range(p + 1, self.n):

                for r in range(q + 1, self.n):

                    for s in range(r + 1, self.n):

                        # If aaaa or bbbb, all three of the following ifs apply, but there are only 3 distinct operators
                        # In the other cases, only one of the ifs applies, 2 distinct operators

                        new_positions = []

                        source_orbs_list = [[r, s], [p, s], [q, s], [p, s], [r, s], [q, s]]
                        target_orbs_list = [[p, q], [q, r], [p, r], [q, r], [p, q], [p, r]]

                        f_operators = [FermionOperator(((t1, 1), (t2, 1), (s1, 0), (s2, 0)))
                                    for (t1, t2), (s1, s2) in zip(source_orbs_list, target_orbs_list)]

                        for i, f_operator in enumerate(f_operators):
                            operator = f_operator - hermitian_conjugated(f_operator)
                            operator = normal_ordered(operator)
                            operator = remove_z_string(operator)
                            pos = self.add_operator(operator, cnots=13, cnot_depth=11,
                                                    source_orbs=source_orbs_list[i], target_orbs=target_orbs_list[i])
                            new_positions.append(pos)

                        new_positions = [pos for pos in new_positions if pos is not None]
                        self._ops_on_qubits[str([p, q, r, s])] = new_positions
                        if self.couple_exchanges:
                            for pos1, pos2 in itertools.combinations(new_positions, 2):
                                self.operators[pos1].twin_string_ops.append(pos2)
                                self.operators[pos2].twin_string_ops.append(pos1)



class CEO(OperatorPool):
    """
    Pool of coupled exchange operators. These operators are linear combinations of qubit excitations. Might be sum,
    difference, or have independent variational parameters.
    CEOs with multiple variational parameters (MVP-CEOs) are not included in this class and must be created using the
    specific MVP_CEO class which inherits from QE. Mixed CEOs, where we decide between using one or multiple variational
    parameters based on the gradients (DVG) or energy (DVE), are contemplated in this class for specific values of class
    constructor arguments. However, they can also be created straightforwardly using the specific classes DVG_CEO and
    DVE_CEO.
    Note that the construction of MVP-CEOs is done on AdaptVQE class. Pool operators here are OVP-CEOs, plus a
    "parent pool" of qubit excitations if we are considering DVG-/DVE-CEOs. In that case, we keep track of which QEs
    appear in each OVP-CEO (which is a linear combination of 4 Pauli strings) in order to have the MVP-CEO acting on the
    same indices readily available.
    """

    name = "CEO"

    def __init__(self,
                 molecule=None,
                 sum=True,
                 diff=True,
                 dvg=False,
                 dve=False,
                 n=None,
                 source_ops=None):
        """
        Arguments:
            molecule (PyscfMolecularData): the molecule for which we will use the pool
            sum (bool): If to consider OVP-CEOs which are sums of QEs.
            diff (bool):  If to consider OVP-CEOs which are differences of QEs.
            dvg (bool): if to consider DVG-CEOs.
            dve (bool): if to consider DVE-CEOs.
            n (int): number of qubits the operator acts on. Note that this includes identity operators - it's dependent
            on system size, not operator support
        """

        # We define the algorithm to start each iteration by selecting one OVP-CEO, thus there have to be such operators
        # in the pool.
        assert sum or diff

        self.sum = sum
        self.diff = diff

        if dvg and "DVG" not in self.name:
            self.name = "DVG-" + self.name
        if dve and "DVE" not in self.name:
            self.name = "DVE-" + self.name

        # We need to keep track of the QEs that are included in the OVP-CEOs, in case we want to "free" the variational
        # parameters and turn them into MVP-CEOs with independent excitations.
        track_parents = dvg or dve
        if track_parents:
            self.parent_pool = QE(molecule, n=n)
        self.track_parents = track_parents

        super().__init__(molecule, n=n, source_ops=source_ops)

        if not diff:
            self.name += "_sum"
        if not sum:
            self.name += "_diff"

    def create_operators(self):
        """
        Create pool operators and insert them into self.operators (list).
        """

        if self.track_parents:
            # Append all parent pool (QE) operators to self. Note that these include singles. They don't
            #yield any linear combination of operators, because there is at most one single per pair of
            #spin-orbitals. Singles belong to the pool independently and individually.
            self.operators = copy(self.parent_pool.operators)
        else:
            self.create_singles()

        self.create_doubles()

    def create_singles(self):
        """
        Create one-body CEOs.
        """

        for p in range(0, self.n):

            for q in range(p + 1, self.n):

                if (p + q) % 2 == 0:
                    f_operator = FermionOperator(((p, 1), (q, 0)))
                    f_operator -= hermitian_conjugated(f_operator)
                    f_operator = normal_ordered(f_operator)

                    q_operator = remove_z_string(f_operator)

                    self.add_operator(q_operator, cnots=2, cnot_depth=2, source_orbs=[q],
                                    target_orbs=[p])

    def create_doubles(self):
        """
        Create one-body CEOs.
        """

        for p in range(0, self.n):

            for q in range(p + 1, self.n):

                for r in range(q + 1, self.n):

                    for s in range(r + 1, self.n):

                        if (p + q + r + s) % 2 != 0:
                            continue

                        if self.track_parents:
                            parents = self.parent_pool.get_ops_on_qubits([p, q, r, s])
                        else:
                            parents = None

                        q_operators, orbs = create_qes(p,q,r,s)

                        for (q_operator_1, q_operator_2), (source_orbs, target_orbs) in zip(q_operators,orbs):

                            if self.sum:
                                self.add_operator(q_operator_1 + q_operator_2, cnots=9, cnot_depth=7, parents=parents,
                                                  source_orbs=source_orbs, target_orbs=target_orbs,
                                                  ceo_type="sum")
                            if self.diff:
                                self.add_operator(q_operator_1 - q_operator_2, cnots=9, cnot_depth=7, parents=parents,
                                                  source_orbs=source_orbs, target_orbs=target_orbs,
                                                  ceo_type="diff")


    @property
    def op_type(self):
        return OpType.QUBIT

    def expm(self, coefficient, index):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient.
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.
        Otherwise, a trigonometric formula leveraging the structure of the operators is used. This is quite faster
            than using generic matrix exponentiation methods.

        Arguments:
            coefficient (float)
            index (int)
        """
        if self.eig_decomp[index] is not None:
            return super().expm(coefficient, index)
        op = self.get_imp_op(index)
        n, n = op.shape
        exp_op = identity(n) + np.sin(coefficient) * op + (1 - np.cos(coefficient)) * self.square(index)
        return exp_op

    def expm_mult(self, coefficient, index, other):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient, multiplying
        another pool operator (indexed "other").
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.
        Otherwise, a trigonometric formula leveraging the structure of the operators is used. This is quite faster
        than using generic matrix exponentiation methods.

        Arguments:
            coefficient (float)
            index (int)
            other (csc_matrix)
        """
        if self.eig_decomp[index] is not None:
            return super().expm_mult(coefficient, index, other)
        '''
        exp_op = self.expm(coefficient, index)
        m = exp_op.dot(other)
        '''
        # It's faster to do product first, then sums; this way we never do matrix-matrix operations, just matrix-vector
        op = self.get_imp_op(index)
        m = op.dot(other)
        # In the following we can use self.square(index).dot(ket) instead of op.dot(m). But that's actually slightly
        # slower even if self.square(index) was already stored so we don't have to calculate it
        m = other + np.sin(coefficient) * m + (1 - np.cos(coefficient)) * op.dot(m)

        return m

    def get_circuit(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments.
        """

        if self.track_parents:
            return self._get_mvp_ceo_circuit(indices, coefficients)
        else:
            return self._get_ovp_ceo_circuit(indices, coefficients)

    def _get_ovp_ceo_circuit(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments.
        Function for OVP-CEO pool only - no DVG- or DVE-CEO.
        """
        circuit = QuantumCircuit(self.n)

        for i, (index, coefficient) in enumerate(zip(indices, coefficients)):
            operator = self.operators[index]
            source_orbs = operator.source_orbs
            target_orbs = operator.target_orbs
            ceo_type = operator.ceo_type
            qc = ovp_ceo_circuit(source_orbs, target_orbs, self.n, coefficient,
                                 ceo_type, big_endian=True)

            circuit = circuit.compose(qc)
            circuit.barrier()

        return circuit

    def _get_mvp_ceo_circuit(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments.
        Function for MVP-CEO pool. Also uses OVP-CEO circuits when those are more hardware-efficient.
        """

        circuit = QuantumCircuit(self.n)
        acc_indices = []
        acc_cs = []

        for i, (index, coefficient) in enumerate(zip(indices, coefficients)):

            if index not in self.parent_range:
                # OVP-CEO
                operator = self.operators[index]
                source_orbs = operator.source_orbs
                target_orbs = operator.target_orbs
                ceo_type = operator.ceo_type
                qc = ovp_ceo_circuit(source_orbs, target_orbs, self.n, coefficient,
                                     ceo_type, big_endian=True)

            else:
                # MVP-CEO
                acc_indices.append(index)
                acc_cs.append(coefficient)

                if (i + 1 < len(indices) and
                        self.get_qubits(indices[i + 1]) == self.get_qubits(index) and
                        indices[i + 1] in self.parent_range):
                    # Wait until you have all QEs with same support to implement
                    # the MVP-CEO circuit
                    qc = None

                elif len(acc_indices) == 1:
                    # Implement as QE (single or double)
                    operator = self.operators[index]
                    source_orbs = operator.source_orbs
                    target_orbs = operator.target_orbs
                    qc = qe_circuit(source_orbs, target_orbs, coefficient, self.n, big_endian=True)
                    acc_indices = []
                    acc_cs = []

                else:
                    # Implement as MVP-CEO
                    q_op = QubitOperator()
                    for ix, c in zip(acc_indices, acc_cs):
                        q_op = q_op + c * self.operators[ix].q_operator
                    qc = mvp_ceo_circuit(q_op, self.n, big_endian=True)
                    acc_indices = []
                    acc_cs = []

            if qc is not None:
                circuit = circuit.compose(qc)
                circuit.barrier()

        return circuit


class OVP_CEO(CEO):
    """
    Pool of coupled exchange operators with one variational parameter.
    """

    name = "OVP_CEO"

    def __init__(self,
                 molecule=None,
                 sum=True,
                 diff=True,
                 n=None):
        super().__init__(molecule=molecule,
                         sum=sum,
                         diff=diff,
                         dvg=False,
                         dve=False,
                         n=n)


class DVG_CEO(CEO):
    """
    Pool of coupled exchange operators with one OR multiple variational parameters. Decision is made using the gradient
    (if the gradient of a QE is nonzero, we add it with an independent variational parameter).
    Note that the construction of MVP-CEOs is done on AdaptVQE class. Pool operators here are OVP-CEOs, plus a
    "parent pool" of qubit excitations. We keep track of which QEs appear in each OVP-CEO (which is a linear combination
    of 4 Pauli strings) in order to have the MVP-CEO acting on the same indices readily available.
    """

    name = "DVG_CEO"

    def __init__(self,
                 molecule=None,
                 sum=True,
                 diff=True,
                 n=None):
        super().__init__(molecule=molecule,
                         sum=sum,
                         diff=diff,
                         dvg=True,
                         dve=False,
                         n=n)


class DVE_CEO(CEO):
    """
    Pool of coupled exchange operators with one OR multiple variational parameters. Decision is made using the energy
    (both options are optimized; the one leading to a higher decrease in energy per unit CNOT is selected).
    Note that the construction of MVP-CEOs is done on AdaptVQE class. Pool operators here are OVP-CEOs, plus a
    "parent pool" of qubit excitations. We keep track of which QEs appear in each OVP-CEO (which is a linear combination
    of 4 Pauli strings) in order to have the MVP-CEO acting on the same indices readily available.
    """

    name = "DVE_CEO"

    def __init__(self,
                 molecule=None,
                 sum=True,
                 diff=True,
                 n=None):
        super().__init__(molecule=molecule,
                         sum=sum,
                         diff=diff,
                         dvg=False,
                         dve=True,
                         n=n)


class MVP_CEO(QE):
    """
    Pool of coupled exchange operators with multiple variational parameters.
    Note that the construction of MVP-CEOs is done on AdaptVQE class. Pool operators here are qubit excitations, just
     like in the QE class; all this pool does is keep track of which operators act on the same indices.
    """

    name = "MVP_CEO"

    def __init__(self,
                 molecule=None,
                 frozen_orbitals=[],
                 n=None,
                 source_ops=None):
        super().__init__(molecule=molecule,
                         couple_exchanges=True,
                         frozen_orbitals=frozen_orbitals,
                         n=n,
                         source_ops=source_ops)

    def get_circuit(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments.
        Function for MVP-CEO pool specifically.
        """
        circuit = QuantumCircuit(self.n)
        acc_indices = []
        acc_cs = []

        for i, (index, coefficient) in enumerate(zip(indices, coefficients)):

            acc_indices.append(index)
            acc_cs.append(coefficient)

            if (i + 1 < len(indices) and
                    self.get_qubits(indices[i + 1]) == self.get_qubits(index)):
                # Wait until you have all QEs with same support to implement
                # the MVP-CEO circuit
                qc = None

            elif len(acc_indices) == 1:
                # Implement as QE (single or double)
                operator = self.operators[index]
                source_orbs = operator.source_orbs
                target_orbs = operator.target_orbs
                qc = qe_circuit(source_orbs, target_orbs, coefficient, self.n, big_endian=True)
                acc_indices = []
                acc_cs = []

            else:
                # Implement as MVP-CEO
                q_op = QubitOperator()
                for ix, c in zip(acc_indices, acc_cs):
                    q_op = q_op + c * self.operators[ix].q_operator
                qc = mvp_ceo_circuit(q_op, self.n, big_endian=True)
                acc_indices = []
                acc_cs = []

            if qc is not None:
                circuit = circuit.compose(qc)
                circuit.barrier()

        return circuit


