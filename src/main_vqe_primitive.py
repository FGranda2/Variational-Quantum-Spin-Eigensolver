# Handle imports
from classiq import * # Using Classiq Version 0.52.0 Release
import numpy as np
from typing import cast, List
import quimb as qu
from functools import reduce
from operator import mul
from classiq.execution import ExecutionSession, ExecutionPreferences
from scipy.optimize import minimize

# Initialization circuit layer
@qfunc
def Init(q: QArray) -> None:
    # First, do PauliX on all qubits
    for i in range(n_bits):
        X(q[i])

    # Next, add the Hadamards and CNOT
    for i in range(0, n_bits, 2):
        hadamard_transform(q[i])
        CX(q[i],q[i + 1])

# N block function
@qfunc
def N_block(q0: QBit, q1: QBit, layer: CReal) -> None:
    RZ(np.pi / 2, q1)

    CX(q1, q0)
     
    RZ(2 * layer - np.pi/2, q0)
    RY(np.pi / 2 - 2 * layer, q1)
    
    CX(q0, q1)

    RY(2 * layer - np.pi/2, q1)

    CX(q1, q0)
    RZ(-np.pi / 2, q0)

# Main ansatz layer
@qfunc
def Sz_conserving_layer(q: QArray, params: CArray[CReal]) -> None:
    # Apply the N_block layer for entangling
    param_idx = 0
    for i in range(0, n_bits, 2):
        N_block(q[i],q[i+1], params[param_idx])
        param_idx += 1

    for i in range(1, n_bits - 1, 2):
        N_block(q[i],q[i+1], params[param_idx])
        param_idx += 1

    # Apply the Phase Gates
    for i in range(n_bits):
        PHASE(params[(n_bits-1)+i], q[i])

# Function to build the Paulis string
def build_pauli_string(N: int, J: float) -> list[tuple[str, float]]:
    pauli_list = []
    
    # Loop over each neighboring pair of spins (i, i+1)
    for i in range(N-1):
        # Create Pauli strings for sigma_x, sigma_y, sigma_z interactions
        for pauli in ['X', 'Y', 'Z']:
            # Identity for other positions, Pauli for position i and i+1
            identity = ['I'] * N
            identity[i] = pauli
            identity[i+1] = pauli
            
            # Join Pauli string and add it to the list with coefficient J
            pauli_string = ''.join(identity)
            pauli_list.append((pauli_string, J))
    
    return pauli_list

# Definition of Pauli Terms
CHAR_TO_STUCT_DICT = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}

# Functions for creating the Hamiltonian from the Paulis String
def pauli_str_to_enums(pauli):
    return [CHAR_TO_STUCT_DICT[s] for s in pauli]

def pauli_list_to_hamiltonian(pauli_list):
    return [
        PauliTerm(
            pauli=pauli_str_to_enums(pauli), coefficient=cast(complex, coeff).real
        )
        for pauli, coeff in pauli_list
    ]

# Circuit design variables
n_bits = 18
n_layers = 5
J = 1
init_strs = ['01' * (n_bits // 2), '10' * (n_bits // 2)]
print(init_strs)
# Create the pauli list
pauli_list = build_pauli_string(n_bits,J)
# Create the Hamiltonian
heis_ham = pauli_list_to_hamiltonian(pauli_list)

param_per_layer = (n_bits - 1)+n_bits
num_parameters = param_per_layer*n_layers

#Defining the Hamiltonian from the problem
HAMILTONIAN = QConstant("HAMILTONIAN", List[PauliTerm], heis_ham)
X0 = list((np.random.rand(num_parameters) - .5) * np.pi)

#Defining the Ansatz for the Problem
@qfunc
def main(q: Output[QArray], p: CArray[CReal, num_parameters]) -> None:
    allocate(n_bits, q)
    # Prepare the initial state
    # [0101....01]
    # for i in range(n_bits):
    #     if i % 2 != 0:
    #         X(q[i])
    # [1010....10]
    for i in range(n_bits):
        if i % 2 != 0:
            X(q[i])

    # Init(q)
    # Do n layers of the Sz_conserving Ansatz
    for i in range(n_layers):
        start_index = i * param_per_layer
        end_index = start_index + param_per_layer
        Sz_conserving_layer(q, p[start_index:end_index])

#Defining the Variational Quantum Eigensolver primitives with proper paramters
@cfunc
def cmain() -> None:
    res = vqe(
        HAMILTONIAN, #Hamiltonian of the problem
        False, #Maximize Parameter
        [],
        optimizer=Optimizer.COBYLA, # Classical Optimizer
        max_iteration=7000,
        tolerance=1e-10,
        step_size=0,
        skip_compute_variance=False,
        alpha_cvar=1,
    )
    save({"result": res})

qmod = create_model(main, classical_execution_function=cmain)
qmod_prefs = set_execution_preferences(
    qmod,
    ExecutionPreferences(num_shots=30000, job_name="N=4, L=5 #"),
)
qprog = synthesize(qmod_prefs)
show(qprog)
write_qmod(qmod_prefs, name="vqe_primitives")

estimation = execute(qprog)
# res.open_in_ide()
vqe_result = estimation.result()[0].value

# print(vqe_result)
print("Minimal energy of the Hamiltonian", vqe_result.energy)
# print("Optimal parameters for the Ansatz", vqe_result.optimal_parameters)