import numpy as np
import matplotlib.pyplot as plt

POPULATION_SIZE = 500
PARAMETER_SIZE = 32
INDIVIDUAL_SIZE = 4 * PARAMETER_SIZE  # 4 Parameter
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.1
GENERATIONS = 100
PARAMETER_RANGE = (-5.0, 5.0)


def gray_to_binary(gray_code: np.array):
    binary = np.zeros_like(gray_code)
    binary[..., 0] = gray_code[..., 0]
    for i in range(1, gray_code.shape[-1]):
        binary[..., i] = binary[..., i - 1] ^ gray_code[..., i]
    return binary


def decode_chromosome(chromosome: np.array) -> tuple:
    params_binary = chromosome.reshape(4, PARAMETER_SIZE)
    params_gray = gray_to_binary(params_binary)

    # Konvertiere Binär zu Integer
    powers_of_2 = 2 ** np.arange(PARAMETER_SIZE - 1, -1, -1, dtype=np.uint64)
    int_values = np.sum(params_gray * powers_of_2, axis=1)

    # Skaliere Integer auf den gewünschten Bereich [-5, 5]
    max_int = 2**PARAMETER_SIZE - 1
    min_val, max_val = PARAMETER_RANGE
    scaled_values = min_val + (int_values / max_int) * (max_val - min_val)

    return tuple(scaled_values)


def fitness(population: np.array) -> np.array:
    fitness_scores = np.zeros(population.shape[0])
    x = np.linspace(-1, 1, 100)
    g_x = np.exp(x)

    for i, individual in enumerate(population):
        a, b, c, d = decode_chromosome(individual)
        f_x = a * x**3 + b * x**2 + c * x + d

        mse = np.mean((f_x - g_x) ** 2)

        # Fitness ist der Kehrwert des Fehlers (plus eine kleine Konstante zur Stabilität)
        fitness_scores[i] = 1.0 / (mse + 1e-9)

    return fitness_scores


# TODO: tournament selection
# def selection(population: np.array, fitnesses: np.array) -> np.array:
#     total_fitness = np.sum(fitnesses)
#     selection_probs = fitnesses / total_fitness

#     # Wähle Indizes für die Eltern basierend auf den Wahrscheinlichkeiten
#     parent_indices = np.random.choice(
#         np.arange(len(population)), size=len(population), p=selection_probs
#     )
#     return population[parent_indices]


def selection(
    population: np.array,
    fitnesses: np.array,
    mode: str = "tournament",
    tournament_size: int = 2,
) -> np.array:
    if mode == "tournament":
        new_population = np.empty_like(population)
        population_size = len(population)
        for i in range(population_size):
            # Select k random individuals for the tournament
            tournament_indices = np.random.choice(
                population_size, size=tournament_size, replace=False
            )
            tournament_fitnesses = fitnesses[tournament_indices]

            # Find the winner of the tournament (the one with the highest fitness)
            winner_index_in_tournament = np.argmax(tournament_fitnesses)
            winner_index_in_population = tournament_indices[winner_index_in_tournament]

            # Add the winner to the new population
            new_population[i] = population[winner_index_in_population]

        return new_population

    total_fitness = np.sum(fitnesses)
    selection_probs = fitnesses / total_fitness

    # Wähle Indizes für die Eltern basierend auf den Wahrscheinlichkeiten
    parent_indices = np.random.choice(
        np.arange(len(population)), size=len(population), p=selection_probs
    )
    return population[parent_indices]


def crossover_and_selection(
    population: np.array, fitnesses: np.array, crossover_rate: float
) -> np.array:
    total_fitness = np.sum(fitnesses)
    if total_fitness == 0:  # Fallback, falls alle Fitness-Werte null sind
        selection_probs = None
    else:
        selection_probs = fitnesses / total_fitness

    new_population = np.empty_like(population)

    for i in range(0, POPULATION_SIZE, 2):
        # Wähle zwei Elternteile
        parent_indices = np.random.choice(
            np.arange(len(population)),
            size=2,
            replace=False,  # Zwei unterschiedliche Eltern wählen
            p=selection_probs,
        )
        parent1, parent2 = population[parent_indices]

        if np.random.rand() < crossover_rate:
            # Wähle einen Crossover-Punkt
            crossover_point = np.random.randint(1, INDIVIDUAL_SIZE - 1)
            # Erzeuge Nachkommen
            new_population[i] = np.concatenate(
                (parent1[:crossover_point], parent2[crossover_point:])
            )
            new_population[i + 1] = np.concatenate(
                (parent2[:crossover_point], parent1[crossover_point:])
            )
        else:
            # Selection
            new_population[i] = parent1
            new_population[i + 1] = parent2

    return new_population


def mutation(population: np.array, mutation_rate: float) -> np.array:
    for i in range(len(population)):
        for j in range(INDIVIDUAL_SIZE):
            if np.random.rand() < mutation_rate:
                population[i, j] = 1 - population[i, j]
    return population


def plot_results(best_params: tuple):
    """Zeichnet die Originalfunktion, die approximierte Funktion und die Taylor-Reihe."""
    a, b, c, d = best_params

    # Taylor-Koeffizienten für e^x um 0 (g(x) ≈ 1 + x + x²/2! + x³/3!)
    # f(x) = ax³ + bx² + cx + d
    # a = 1/6, b = 1/2, c = 1, d = 1
    taylor_coeffs = (1 / 6, 1 / 2, 1, 1)

    print("\n--- Vergleich der Koeffizienten ---")
    print(f"         \t{'a (x³)':>10}\t{'b (x²)':>10}\t{'c (x)':>10}\t{'d (1)':>10}")
    print(f"Gen. Alg.:\t{a:10.6f}\t{b:10.6f}\t{c:10.6f}\t{d:10.6f}")
    print(
        f"Taylor:  \t{taylor_coeffs[0]:10.6f}\t{taylor_coeffs[1]:10.6f}\t{taylor_coeffs[2]:10.6f}\t{taylor_coeffs[3]:10.6f}"
    )

    x = np.linspace(-1, 1, 200)
    g_x = np.exp(x)
    f_x_ga = a * x**3 + b * x**2 + c * x + d
    f_x_taylor = (
        taylor_coeffs[0] * x**3
        + taylor_coeffs[1] * x**2
        + taylor_coeffs[2] * x
        + taylor_coeffs[3]
    )

    plt.figure(figsize=(10, 6))
    plt.plot(x, g_x, label="Original: $g(x) = e^x$", linewidth=3, color="black")
    plt.plot(
        x, f_x_ga, label="Approximation (Gen. Algorithmus)", linestyle="--", color="red"
    )
    plt.plot(
        x, f_x_taylor, label="Approximation (Taylor-Reihe)", linestyle=":", color="blue"
    )
    plt.title("Approximation von $e^x$ im Bereich [-1, 1]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 3)
    plt.show()


def main():
    # 1. Initialisierung
    population = np.random.randint(0, 2, size=(POPULATION_SIZE, INDIVIDUAL_SIZE))
    best_individual = None
    best_fitness = -1

    print("Starte genetischen Algorithmus...")
    for generation in range(GENERATIONS):
        # 1. Fitness-Berechnung
        fitness_scores = fitness(population)

        # Bestes Individuum der Generation speichern
        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_fitness:
            best_fitness = fitness_scores[current_best_idx]
            best_individual = population[current_best_idx]

        if (generation + 1) % 10 == 0:
            print(
                f"Generation {generation+1}/{GENERATIONS} | Beste Fitness: {best_fitness:.4f}"
            )

        # 2. Selection & Crossover
        offspring = crossover_and_selection(population, fitness_scores, CROSSOVER_RATE)

        # 3. Mutation
        new_population = mutation(offspring, MUTATION_RATE)

        # Sicherstellen, dass das beste Individuum überlebt
        new_population[0] = best_individual

        population = new_population

    print("\nOptimierung abgeschlossen.")
    best_params = decode_chromosome(best_individual)
    plot_results(best_params)


if __name__ == "__main__":
    main()
