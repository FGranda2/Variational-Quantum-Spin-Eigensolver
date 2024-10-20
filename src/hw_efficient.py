# Handle imports
from classiq import * # Using Classiq Version 0.52.0 Release
import numpy as np
from typing import cast, List
from classiq.execution import ExecutionPreferences

# Main ansatz layer
@qfunc
def hw_efficient_layer(q: QArray, params: CArray[CReal]) -> None:
    # Apply the rotations first
    param_idx = 0

    # Start with the RY's
    for i in range(n_bits):
        RY(params[param_idx], q[i])
        param_idx += 1

    # Continue with the RZ's
    for i in range(n_bits):
        RZ(params[param_idx], q[i])
        param_idx += 1

    # Apply the Hadamard/CNOT chains
    for i in range(0,n_bits-1):
        hadamard_transform(q[i+1])
        CX(q[i],q[i+1])
        hadamard_transform(q[i+1])

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
n_bits = 4
n_layers = 5
J = 1
JOB_NAME = "".join(("HW_efficient | ","N_bits:",str(n_bits)," Layers:",str(n_layers)))

# Create the pauli list
pauli_list = build_pauli_string(n_bits,J)
# Create the Hamiltonian
heis_ham = pauli_list_to_hamiltonian(pauli_list)

# Determine the # of parameters
param_per_layer = n_bits * 2 # The RY's and RZ's
num_parameters = param_per_layer*n_layers

# Defining the Hamiltonian from the problem
HAMILTONIAN = QConstant("HAMILTONIAN", List[PauliTerm], heis_ham)

# Defining the initial parameter values
X0 = list((np.random.rand(num_parameters) - .5) * np.pi)
INITIAL_POINT = QConstant("INITIAL_POINT", List[float],X0)

# Defining the Ansatz for the Problem
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

    # Do n layers of the Sz_conserving Ansatz
    for i in range(n_layers):
        start_index = i * param_per_layer
        end_index = start_index + param_per_layer
        hw_efficient_layer(q, p[start_index:end_index])

# Defining the Variational Quantum Eigensolver p4rimitives with proper paramters
@cfunc
def cmain() -> None:
    res = vqe(
        hamiltonian=HAMILTONIAN,
        maximize=False,
        initial_point=INITIAL_POINT,
        optimizer=Optimizer.NELDER_MEAD, # Classical Optimizer
        max_iteration=10000,
        tolerance=1e-10,
        step_size=0,
        skip_compute_variance=False,
        alpha_cvar=1,
    )
    save({"result": res})

# Model, preferences and synthesize
qmod = create_model(main, classical_execution_function=cmain)
qmod_prefs = set_execution_preferences(
    qmod,
    ExecutionPreferences(num_shots=10000, job_name=JOB_NAME),
)
qprog = synthesize(qmod_prefs)
show(qprog)
write_qmod(qmod_prefs, name="vqe_primitives")

# Execution
estimation = execute(qprog)
vqe_result = estimation.result()[0].value

print("Minimal energy of the Hamiltonian", vqe_result.energy)
# print("Optimal parameters for the Ansatz", vqe_result.optimal_parameters)