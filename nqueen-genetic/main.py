import random
import matplotlib.pyplot as plt

# calculate the conflict numbers of each solution
def fitness_score(chromosome):
    n = len(chromosome)
    diagonal_collisions = 0
    horizontal_collisions = sum(
        [chromosome.count(queen)-1 for queen in chromosome])/2

    # Diagonal conflicts
    for i in range(n):
        for j in range(i+1, n):
            if abs(i-j) == abs(chromosome[i] - chromosome[j]):
                diagonal_collisions += 1

    score = int((n*(n-1))/2 - (horizontal_collisions + diagonal_collisions))

    return score


def selection(population):
    fitness_values = [fitness_score(chromosome) for chromosome in population]
    total_fitness = sum(fitness_values)
    probabilities = [fitness_value /
                     total_fitness for fitness_value in fitness_values]
    parents = random.choices(population, weights=probabilities, k=2)

    return parents


def crossover(parents):
    n = len(parents[0])
    crossover_point = random.randint(1, n-1)
    offspring = parents[0][:crossover_point] + parents[1][crossover_point:]

    return offspring


def mutation(chromosome, mutation_probability):
    n = len(chromosome)
    mutated_chromosome = chromosome[:]

    for i in range(n):
        if random.random() < mutation_probability:
            mutated_chromosome[i] = random.randint(0, n-1)

    return mutated_chromosome


def genetic_algorithm(population_size, mutation_probability, max_generations, n):
    population = [[random.randint(0, n-1) for i in range(n)]
                  for j in range(population_size)]
    
    
    fitness_history = []
    best_solutions = []

    for generation in range(max_generations):
        new_population = []

        for i in range(population_size//2):
            parents = selection(population)
            offspring = crossover(parents)
            offspring = mutation(offspring, mutation_probability)
            new_population.extend([parents[0], parents[1], offspring])

        population = new_population

        # if there is solution
        for chromosome in population:
            if fitness_score(chromosome) == (n*(n-1))/2:
                best_solutions.append(chromosome)
                return chromosome, fitness_history, best_solutions

        # calculate average score to plot them and store best solutions so far in every generate
        avg_fitness = sum([fitness_score(chromosome)
                          for chromosome in population])/population_size
        fitness_history.append(avg_fitness)
        best_solution_so_far = max(population, key=fitness_score)
        best_solutions.append(best_solution_so_far)

    # If there was no solution
    return None, fitness_history, best_solutions

# plot solution board


def plot_board(chrom):
    nq = len(chrom)
    board = []

    for x in range(nq):
        board.append(["x"] * nq)

    for i in range(nq):
        board[chrom[i]][i] = "Q"

    fig, ax = plt.subplots()
    ax.imshow([[0.5 if (i + j) % 2 == 0 else 1 for i in range(nq)]
              for j in range(nq)], cmap='binary', interpolation='nearest')
    for i in range(nq):
        for j in range(nq):
            if board[i][j] == 'Q':
                ax.text(j, i, '♕', fontsize=20,
                        ha='center', va='center', color='red')

    ax.set_title('Solution for {} Queens Problem'.format(nq))
    plt.show()


# set number of queens and start genetic algorithm
n = 8
solution, fitness_history, best_solutions = genetic_algorithm(
    population_size=100, mutation_probability=0.1, max_generations=500, n=n)
print(len(best_solutions))
if solution is not None:
    plot_board(solution)
else:
    print("no answer")

# plot average fitness score over generations
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Average Fitness Score')
plt.title('Fitness Progression over Generations')
plt.show()

# plot best soloutions so far for every generate
num_plots = len(best_solutions)
rows = (num_plots-1) // 9 + 1

for i in range(rows):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    for j in range(9):
        if i*9+j < num_plots:
            row_idx = j // 3
            col_idx = j % 3

            axs[row_idx, col_idx].imshow([[0.5 if (x + y) % 2 == 0 else 1 for x in range(n)]
                                         for y in range(n)], cmap='binary', interpolation='nearest')
            chrom = best_solutions[i*9+j]
            board = []
            for x in range(n):
                board.append(["x"] * n)
            for k in range(n):
                board[chrom[k]][k] = "Q"
            for k in range(n):
                axs[row_idx, col_idx].text(
                    k, chrom[k], '♕', fontsize=20, ha='center', va='center', color='red')
            axs[row_idx, col_idx].set_title('Generation {}'.format(i*9+j+1))
            axs[row_idx, col_idx].axis('off')

    plt.show()
