import numpy as np
import pandas as pd
import yfinance as yf

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


def build_portfolio_ising_hamiltonian(
    tickers, start_date, end_date, q=0.5, budget=2, penalty=5.0
):
    """
    Download historical prices and construct the Ising Hamiltonian associated
    with a Markowitz portfolio-selection problem under a cardinality constraint.

    - q: risk aversion parameter.
    - budget: number of assets to include in the portfolio.
    - penalty: penalty strength enforcing the budget constraint.
    """
    # Download adjusted close prices.
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()

    # Compute expected returns and annualized covariance matrix.
    mu = returns.mean().values * 252
    sigma = returns.cov().values * 252
    num_assets = len(tickers)

    # Build the quadratic optimization problem in Qiskit.
    qp = QuadraticProgram(name="Portfolio_Optimization")

    # Binary variables x_i for each asset.
    for i in range(num_assets):
        qp.binary_var(name=f"x_{i}")

    # Markowitz objective: q * x^T * Sigma * x - mu^T * x.
    linear = -mu
    quadratic = q * sigma

    # Budget constraint: sum_i x_i = budget.
    linear_constraint = {f"x_{i}": 1.0 for i in range(num_assets)}
    """
            Defines the constraint followed by the model. In this case, the left hand side is defined
            above in linear_constraint, the mathematical relation is equality, and the right hand side (rhs below)
            is the budget, which is the number of assets to include in the portfolio.
            """
    qp.linear_constraint(
        linear=linear_constraint, sense="==", rhs=budget, name="budget_constraint"
    )

    # Minimize the defined cost function.
    qp.minimize(linear=linear, quadratic=quadratic)

    # Convert the problem to a QUBO with a quadratic penalty term. Please refer to README for details.
    converter = QuadraticProgramToQubo(penalty=penalty)
    qubo = converter.convert(qp)

    # Map the QUBO into an Ising Hamiltonian (Pauli Z operators).
    operator, offset = qubo.to_ising()

    print(f"Module 1")
    print(f"Number of qubits (assets): {num_assets}")

    # Print the ising operator in two list: the first shows the operators, the second the coefficients.
    print(f"Ising operator generated: \n{operator}")

    return operator, num_assets, qubo, data


# Example usage with four S&P 500 assets
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
H_C, N, qubo_problem, market_data = build_portfolio_ising_hamiltonian(
    tickers=tickers, start_date="2023-01-01", end_date="2024-01-01", q=0.5, budget=2
)

from math import comb
from itertools import combinations
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler


def create_qaoa_circuit(num_qubits, cost_operator, gamma, beta):
    """
    Construct a single-layer QAOA circuit for the cost Hamiltonian.
    """
    qc = QuantumCircuit(num_qubits)

    # Initial state: |+>^{\otimes N}. Based on the chosen initial state for the adiabatic theorem.
    qc.h(range(num_qubits))

    # Cost evolution U(H_C, gamma): apply Z and ZZ rotations according to each Pauli term.
    for pauli, coeff in zip(
        cost_operator.paulis, cost_operator.coeffs
    ):  # join the lists of operator labels and coefficients.
        real_coeff = (
            coeff.real
        )  # Qiskit uses complex numbers for coefficients, but this program does not generate any.
        qubit_indices = np.where(pauli.z)[
            0
        ]  # pauli.z is a boolean array indicating which qubits are acted on by Z.

        if len(qubit_indices) == 1:  # divide between single and double qubit operators.
            idx = qubit_indices[0]
            qc.rz(2 * gamma * real_coeff, idx)
        elif len(qubit_indices) == 2:
            i, j = qubit_indices
            qc.rzz(2 * gamma * real_coeff, i, j)

    # Mixer evolution U(H_B, beta): apply X rotations on each qubit.
    for i in range(num_qubits):
        qc.rx(-2 * beta, i)

    qc.measure_all()
    return qc


def extract_observables_vector(probabilities, num_qubits):
    """
    Extracts the observable vector y_k = [<Z_i>, <Z_i Z_j>] directly
    from the probability dictionary (probabilities) {bitstring: prob}.
    """
    # 1. Calculate <Z_i>
    z_single = np.zeros(
        num_qubits
    )  # empty array to store the expected values of each qubit's Z operator.

    # 2. Calculate <Z_i Z_j>
    pairs = list(combinations(range(num_qubits), 2))
    """
    Pairs is a list of all possible combinations of qubit indices withou repetition.
    """
    z_pairs = np.zeros(
        len(pairs)
    )  # Empty array to store the expected values of each pair of qubits' ZZ operator.

    for bitstring, prob in probabilities.items():
        spin_values = np.array(
            [1.0 if bitstring[-(i + 1)] == "0" else -1.0 for i in range(num_qubits)]
        )
        """
        spin_values is an array that converts the bitstring into spin values: 0 -> +1, 1 -> -1.
        Please note the little-endian ordering: the rightmost bit corresponds to qubit 0,
        thus choosing bitstring[-(i + 1)] to access the bits in the correct order.
        """
        # Add each value to the corresponding observable expectation value.
        z_single += spin_values * prob
        """
        z_single is updated by adding the spin values multiplied by the probability of the current bitstring.
        This will be performed for each iteration over the probabilities, effectively computing the 
        expected value of each qubit's Z operator across all measured bitstrings.
        """
        # For each pair of qubits, compute the expected value of the ZZ operator.
        for idx, (i, j) in enumerate(pairs):
            z_pairs[idx] += (spin_values[i] * spin_values[j]) * prob
        """
        For each iteration of the most outermost loop (the one that iterates over the probabilities),
        the inner loop iterates over all pairs of qubits. For each pair (i,j), it computes the product
        of the spin values for qubits i and j, then multiplies it by the probability of the current bitstring.
        This value is added to the corresponding index in z_pairs, which will eventually hold the expected value for the ZZ operator.
        """
    # Concatenate to form the characters vector y_k = [<Z_i>, <Z_i Z_j>].
    y_k = np.concatenate([z_single, z_pairs])
    return y_k


sampler = StatevectorSampler()
test_circuit = create_qaoa_circuit(num_qubits=N, cost_operator=H_C, gamma=0.5, beta=0.8)

# Execute the circuit.
job = sampler.run([test_circuit], shots=1000)
result = job.result()[0]

# Retrieve the measurement distribution directly.
counts = result.data.meas.get_counts()
total_shots = sum(counts.values())
probabilities = {bitstring: count / total_shots for bitstring, count in counts.items()}

# Extract the observable vector y_k.
y_k = extract_observables_vector(probabilities, num_qubits=N)

print(f"\nModule 2")
print(f"Dimension of y_k: {len(y_k)} (expected: {N + comb(N,2)})")
print(f"Computed y_k vector: \n{np.round(y_k, 4)}")

from qiskit.circuit.library import EfficientSU2, ZZFeatureMap
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp


def build_hardware_efficient_vqc(input_dim, reps_data=1, reps_ansatz=2, maxiter=100):
    """Construct a compact VQC with a ZZ feature map and an EfficientSU2 ansatz."""
    # Feature map for encoding the QAOA observables.
    feature_map = ZZFeatureMap(  # chosen for data encoding due to its ability to capture correlations between features.
        feature_dimension=input_dim,
        reps=reps_data,
        entanglement="linear",
        parameter_prefix="x",
    )

    # Trainable variational ansatz.
    ansatz = EfficientSU2(  # chosen for its hardware-efficient structure and ability to represent complex quantum states.
        num_qubits=input_dim,
        su2_gates=["ry", "rz"],
        entanglement="linear",
        reps=reps_ansatz,
        insert_barriers=True,
        parameter_prefix="w",
    )

    input_params = list(feature_map.parameters)
    weight_params = list(ansatz.parameters)
    vqc_circuit = feature_map.compose(ansatz)

    estimator = StatevectorEstimator()

    # A separate regressor is trained for each component of the descent direction: gamma and beta.
    def z_observable(
        n_qubits, qubit_idx
    ):  # a helper function to create a Z observable for a specific qubit.
        label = ["I"] * n_qubits
        label[n_qubits - 1 - qubit_idx] = (
            "Z"  # set the Pauli-Z operator for the specified qubit following the little-endian convention.
        )
        return SparsePauliOp("".join(label))

    vqc_models = []
    for qubit_idx in (0, 1):
        qnn = EstimatorQNN(  # turns our previous circuit into a QNN Estimator
            circuit=vqc_circuit,
            estimator=estimator,
            observables=z_observable(input_dim, qubit_idx),
            input_params=input_params,
            weight_params=weight_params,
        )
        vqc_models.append(  # finally creates a Regressor as a Neural Network following the previously developed circuit
            NeuralNetworkRegressor(
                neural_network=qnn,
                optimizer=COBYLA(maxiter=maxiter),
                warm_start=True,  # allows the regressor to use previous data to start from a better position
            )
        )

    ops_count = vqc_circuit.count_ops()
    cnot_count = ops_count.get("cx", 0) + ops_count.get("cz", 0)

    print("Module 3")
    print(f"Qubits in use: {input_dim}")
    print(f"Input parameters (Data): {len(input_params)}")
    print(f"Trainable parameters (Weights): {len(weight_params)}")
    print(f"Estimated circuit depth: {vqc_circuit.depth()}")
    print(f"Entangling gates (CNOTs/CZs): {cnot_count}")

    return vqc_models


def compute_energy_expectation(counts, cost_operator, num_qubits):
    """Compute the expected energy <H_C> from a set of measurement outcomes."""
    total_shots = sum(counts.values())
    energy = 0.0

    for term, coeff in cost_operator.to_list():
        pauli_str = str(term)
        qubit_indices = [
            i for i, char in enumerate(reversed(pauli_str)) if char == "Z"
        ]  # reversed to follow the little-endian convention
        real_coeff = coeff.real

        term_expectation = 0.0  # initialize the expectation of the current term to zero
        for bitstring, count in counts.items():
            prob = count / total_shots
            rev_bitstring = bitstring[
                ::-1
            ]  # reverse the bitstring to align with the qubit indices in little-endian

            val = 1.0  # initialize the eigenvalue of the current term as 1.
            for idx in qubit_indices:
                val *= 1.0 if rev_bitstring[idx] == "0" else -1.0
            term_expectation += val * prob
            """
            This loop iterates over the qubit indices that are acted upon by the current Pauli term.
            The outer loop iterates over the bitstrings (states). This innermost loop is responsible of
            calculating the eigenvalue of the current Pauli term for the given bitstring by the outer loop.
            For each term within the cost Hamiltonian, it will obtain an eigenvalue that will add up as "the cost"
            of using that state.

            """
        energy += (
            real_coeff * term_expectation
        )  # the cost of that state is multiplied by the amount of times that state appears.

    return energy  # it returns the expected energy of the cost Hamiltonian for the given measurement outcomes.


def estimate_energy_gradient(
    gamma, beta, cost_operator, num_qubits, sampler, shots=1000, eps=0.05
):
    """Estimate the energy gradient with central finite differences."""

    def energy_at(g, b):
        qc = create_qaoa_circuit(num_qubits, cost_operator, g, b)
        job = sampler.run([qc], shots=shots)
        result = job.result()[0]
        counts = result.data.meas.get_counts()
        probs = {bit: cnt / shots for bit, cnt in counts.items()}
        return compute_energy_expectation(probs, cost_operator, num_qubits)

    E_g_plus = energy_at(gamma + eps, beta)
    E_g_minus = energy_at(gamma - eps, beta)
    grad_gamma = (E_g_plus - E_g_minus) / (2 * eps)

    E_b_plus = energy_at(gamma, beta + eps)
    E_b_minus = energy_at(gamma, beta - eps)
    grad_beta = (E_b_plus - E_b_minus) / (2 * eps)

    # The descent direction is the negative gradient.
    return np.array([-grad_gamma, -grad_beta])


def run_qaoa_vqc_optimization(cost_operator, num_qubits, steps, lr):
    pairs = list(combinations(range(num_qubits), 2))
    input_dim = num_qubits + len(pairs)

    vqc_models = build_hardware_efficient_vqc(input_dim, maxiter=20)

    gamma_k = np.random.uniform(
        0, np.pi
    )  # initialize gamma_k randomly in the range [0, pi]
    beta_k = np.random.uniform(
        0, np.pi / 2
    )  # initialize beta_k randomly in the range [0, pi/2]
    energy_history = []
    param_history = [(gamma_k, beta_k)]
    sampler = StatevectorSampler()

    for step in range(steps):
        qc = create_qaoa_circuit(num_qubits, cost_operator, gamma_k, beta_k)
        shots = 1000
        job = sampler.run([qc], shots=shots)
        result = job.result()[0]
        counts_formatted = result.data.meas.get_counts()
        probabilities = {bit: cnt / shots for bit, cnt in counts_formatted.items()}

        E_k = compute_energy_expectation(probabilities, cost_operator, num_qubits)
        y_k = extract_observables_vector(probabilities, num_qubits)
        energy_history.append(E_k)
        print(
            f"Step {step+1:02d}/{steps:02d} | Energy <H_C>: {E_k:.4f} | gamma: {gamma_k:.3f}, beta: {beta_k:.3f}"
        )

        vqc_input = y_k.reshape(
            1, -1
        )  # y_k is a vector, but the VQC expects a 2D array. Thus y_k is turn into a one row matrix.

        # Reference target: true descent direction estimated via finite differences.
        true_direction = estimate_energy_gradient(
            gamma_k, beta_k, cost_operator, num_qubits, sampler
        )

        # Each regressor learns one scalar component of the direction vector.
        delta = np.empty(2)
        for idx, model in enumerate(vqc_models):
            model.fit(
                vqc_input, np.array([true_direction[idx]])
            )  # trains the model based on the current energy gradient and state of the QAOA
            delta[idx] = float(
                np.asarray(model.predict(vqc_input)).reshape(-1)[0]
            )  # the model predicts the best change in gamma or beta

        delta_gamma = np.tanh(delta[0]) * lr
        delta_beta = np.tanh(delta[1]) * lr

        # gamma_k and beta_k are updated for the next iteration (and re-scaled due to periodicity)
        gamma_k = (gamma_k + delta_gamma) % np.pi
        beta_k = (beta_k + delta_beta) % (np.pi / 2)
        param_history.append((gamma_k, beta_k))

    return energy_history, param_history, counts_formatted


# Final interpretation


def interpret_portfolio_solution(counts_formatted, tickers):
    """Translate the best QAOA measurement into a selected portfolio."""
    best_bitstring = max(counts_formatted, key=counts_formatted.get)
    max_counts = counts_formatted[best_bitstring]
    total_shots = sum(counts_formatted.values())
    probability = (max_counts / total_shots) * 100

    print("        FINAL INVESTMENT RESULT            ")
    print(
        f"Most probable measured bitstring: '{best_bitstring}' (Confidence: {probability:.1f}%)"
    )
    print("Recommended portfolio allocation:\n")

    selected_assets = []

    # Qiskit uses little-endian ordering; reverse the bitstring to align it with the ticker list.
    for idx, bit in enumerate(reversed(best_bitstring)):
        status = "INCLUDE" if bit == "1" else "EXCLUDE"
        symbol = tickers[idx] if idx < len(tickers) else f"Asset_{idx}"
        print(f"  • {symbol:10s} : [{status}]")

        if bit == "1":
            selected_assets.append(symbol)

    print(
        f"\n -> Selected portfolio: {selected_assets if selected_assets else 'No assets selected (empty portfolio)'}"
    )
    return selected_assets, best_bitstring


# Run the hybrid QAOA + VQC experiment.
energies, params, final_counts = run_qaoa_vqc_optimization(H_C, N, steps=30, lr=0.15)

# Interpret the final optimal solution.
selected_portfolio, optimal_bitstring = interpret_portfolio_solution(
    final_counts, tickers
)
