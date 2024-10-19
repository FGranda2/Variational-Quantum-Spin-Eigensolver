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
            pauli_list.append((pauli_string, -J))
    
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
n_layers = 15
J = 1

# Create the pauli list
pauli_list = build_pauli_string(n_bits,J)
# Create the Hamiltonian
heis_ham = pauli_list_to_hamiltonian(pauli_list)

param_per_layer = (n_bits - 1)+n_bits
num_parameters = param_per_layer*n_layers

@qfunc
def main(q: Output[QArray], p: CArray[CReal, num_parameters]) -> None:
    # Create the array of bits of proper size
    # q = QArray("q")
    allocate(n_bits, q)

    # Prepare the initial state
    # for i in range(n_bits):
    #     if i % 2 != 0:
    #         X(q[i]) 
    
    # Do the initialization circuit
    Init(q)

    # Do n layers of the Sz_conserving Ansatz
    for i in range(n_layers):
        start_index = i * param_per_layer
        end_index = start_index + param_per_layer
        Sz_conserving_layer(q, p[start_index:end_index])


# Simulation
mod = create_model(main)
qprog = synthesize(mod)
show(qprog)

# Start Session
execution_session = ExecutionSession(qprog, execution_preferences=ExecutionPreferences(num_shots=100000, job_name="Job"))
x0 = 2 * np.pi * np.random.random(num_parameters)

# Create dict for results
cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

# Define the cost function and update dict
def cost_function(params, session, hamiltonian):
    
    estimate_result = session.estimate(hamiltonian, {"p": list(params)})
    # print("Current Result: ", estimate_result)
    energy = estimate_result.value.real
    # Update cost history
    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")
    print(params)
    print("-----------------------------------------------")
    return energy

# Minimize and print results
res = minimize(
        cost_function,
        x0,
        args=(execution_session, heis_ham),
        method='Nelder-Mead',
        tol=1e-10,
        options={'disp': False}  # Set maximum iterations to 5000
    )

print(res)