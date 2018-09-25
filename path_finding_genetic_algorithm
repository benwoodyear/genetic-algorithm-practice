import random
import time
import matplotlib.pyplot as plt
import numpy as np


def create_ceiling(matrix):
    """
    Sums the max value from each row to create a total to assess fitness against.
    """
    rows = np.size(matrix, 0)
    ceiling = 0
    for i in range(rows):
        ceiling += max(matrix[i, :])
    return ceiling


def fitness(matrix, test_path):
    """
    This gives a path through the matrix a fitness score, this is normalised by dividing by the maximal possible
    score. This score is then used to rank the different paths allowing the best ones to be selected for reproduction.
    """
    score = 0
    rows = np.size(matrix, 0)
    position = 0
    for i in range(rows):
        position = position + test_path[i]
        score += matrix[i, position]
    return score * 100 / create_ceiling(matrix)


def path_sum(matrix, test_path):
    """
    This calculates the total sum of a path through a matrix, by summing all the values in the locations specified
    by the path.
    """
    score = 0
    rows = np.size(matrix, 0)
    position = 0
    for i in range(rows):
        position = position + test_path[i]
        score += matrix[i, position]
    return score


def generate_path(matrix):
    """
    Generates a random path long enough to navigate from the top to the bottom of the matrix. In this a 0 corresponds
    to moving straight down, and a 1 to moving down and right.
    """
    # Note that the initial 0 is only for starting in the top left
    rows = np.size(matrix, 0)
    path = [0]
    for i in range(rows - 1):
        path.append(random.getrandbits(1))
    return path


def generate_first_population(population_size, matrix):
    """
    Returns a list containing a population of randomly generated paths.
    """
    population = []
    for i in range(population_size):
        population.append(generate_path(matrix))
    return population


def sorted_population(population, matrix):
    """
    Takes the population and the matrix, and works out which paths correspond to the highest scores. This returns a
    N x 2 matrix with the paths in the first column and the corresponding fitness in the second. The returned matrix
    is sorted by fitness, with highest fitness at the top.
    """
    N = len(population)
    path_fitness_matrix = np.zeros([N, 2], dtype=object)
    for i in range(N):
        individual = population[i]
        path_fitness_matrix[i, 0] = individual
        path_fitness_matrix[i, 1] = fitness(matrix, individual)
    sorted_pop = path_fitness_matrix[path_fitness_matrix[:, 1].argsort()[::-1]]
    return sorted_pop


def choose_parents(ordered_population, number_of_best, number_of_random):
    """
    Decides which paths will be chosen to produce children. A certain number of the best paths are chosen then some
    are also selected randomly. This random selection is so that the population doesn't get stuck in a local rather
    than global maximum.
    """
    next_generation = []
    for i in range(number_of_best):
        next_generation.append(ordered_population[i, 0])
    for i in range(number_of_random):
        next_generation.append(random.choice(ordered_population)[0])
    random.shuffle(next_generation)
    return next_generation


def create_child(individual1, individual2):
    """
    This is where the 'genetic' part of the algorithm comes in. The genes for two parents are randomly combined to
    create a child.
    """
    child = []
    for i in range(len(individual1)):
        if random.random() < 0.5:
            child.append(individual1[i])
        else:
            child.append(individual2[i])
    return child


def create_children(breeders, number_of_children):
    """
    Takes the list of parents to breed and the number of children each couple will produce and from this makes the
    next generation. Note that the number of children should be chosen so that the population remains at the same size
    after reproduction.
    """
    next_population = []
    for i in range(int(len(breeders) / 2)):
        for j in range(number_of_children):
            next_population.append(create_child(breeders[i], breeders[len(breeders) - 1 - i]))
    return next_population


def mutate_path(path):
    """
    Randomly mutates one bit of a path. Like the random selection of some parents this is to prevent the population
    becoming stuck in a sub-optimal route. For longer paths may be worth increasing the number of mutations which
    occur.
    """
    index_modification = int(random.random() * len(path))
    # Don't want to modify the first bit, as all paths start in the top left.
    if index_modification != 0:
        path[index_modification] = random.getrandbits(1)
    else:
        pass
    return path


def mutate_population(population, mutation_chance):
    """
    Mutates some members of the population.
    """
    for i in range(len(population)):
        if random.random() * 100 < mutation_chance:
            population[i] = mutate_path(population[i])
        else:
            pass
    return population


def make_next_generation(initial_generation, matrix, top_sample, random_sample, number_of_children, mutation_chance):
    ordered_population = sorted_population(initial_generation, matrix)
    next_breeders = choose_parents(ordered_population, top_sample, random_sample)
    non_mutated_pop = create_children(next_breeders, number_of_children)
    next_generation = mutate_population(non_mutated_pop, mutation_chance)
    return next_generation


def multiple_generations(number_of_generations, matrix, size_population, best_sample, lucky_few, number_of_child,
                         chance_of_mutation):

    lineage = [generate_first_population(size_population, matrix)]
    for i in range(number_of_generations):
        lineage.append(
            make_next_generation(lineage[i], matrix, best_sample, lucky_few, number_of_child, chance_of_mutation))
    return lineage


# Print result:
def simple_result(lineage, matrix, number_of_generations):
    """
    Finds the best solution in historic. Caution not the last result.
    """
    result = best_individuals_from_lineage(lineage, matrix)[number_of_generations - 1]
    return "solution:", path_sum(matrix, result[0]), result[0]


# Analysis tools
def find_best_individual(population, matrix):
    return sorted_population(population, matrix)[0]


def best_individuals_from_lineage(lineage, matrix):
    best_individuals = []
    for population in lineage:
        best_individuals.append(find_best_individual(population, matrix))
    return best_individuals


# graph
def evolution_best_fitness(lineage, matrix):
    plt.axis([0, len(lineage), 0, 105])
    plt.title('Best fitness from each generation')
    evolution_fitness = []
    for population in lineage:
        evolution_fitness.append(find_best_individual(population, matrix)[1])
    plt.plot(evolution_fitness)
    plt.ylabel('Fitness  of best individual')
    plt.xlabel('Generation')
    plt.show()


def evolution_average_fitness(historic, matrix, size_population):
    plt.axis([0, len(historic), 0, 105])
    plt.title('Average fitness of generations')
    evolution_fitness = []
    for population in historic:
        ordered_population = sorted_population(population, matrix)
        average_fitness = 0
        for individual in ordered_population:
            average_fitness += individual[1]
        evolution_fitness.append(average_fitness / size_population)
    plt.plot(evolution_fitness)
    plt.ylabel('Average fitness')
    plt.xlabel('Generation')
    plt.show()


# Function to turn text file into a matrix
def text_to_matrix(file):
    """
    This takes a pyramid of numbers in a txt file and converts it to a square numpy array, with blank values replaced
    by zeroes. np.loadtxt() might be quicker, but this also inputs the zeros.
    """
    values_raw = open(file, 'r')
    values_clean = [[int(x) for x in line.split()] for line in values_raw.readlines()]
    size = len(values_clean)
    matrix = np.zeros([size, size])
    for i in range(size):
        for j in range(i+1):
            matrix[i, j] = values_clean[i][j]
    return matrix


# Small matrix to test
small_test_matrix = np.array([[3, 0, 0, 0], [7, 4, 0, 0], [2, 4, 6, 0], [8, 5, 9, 3]])

# Slightly larger matrix
medium_test_matrix = np.array([[75,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [95, 64,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [17, 47, 82,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [18, 35, 87, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [20,  4, 82, 47, 65,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [19,  1, 23, 75,  3, 34,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [88,  2, 77, 73,  7, 63, 67,  0,  0,  0,  0,  0,  0,  0,  0],
                            [99, 65,  4, 28,  6, 16, 70, 92,  0,  0,  0,  0,  0,  0,  0],
                            [41, 41, 26, 56, 83, 40, 80, 70, 33,  0,  0,  0,  0,  0,  0],
                            [41, 48, 72, 33, 47, 32, 37, 16, 94, 29,  0,  0,  0,  0,  0],
                            [53, 71, 44, 65, 25, 43, 91, 52, 97, 51, 14,  0,  0,  0,  0],
                            [70, 11, 33, 28, 77, 73, 17, 78, 39, 68, 17, 57,  0,  0,  0],
                            [91, 71, 52, 38, 17, 14, 91, 43, 58, 50, 27, 29, 48,  0,  0],
                            [63, 66,  4, 68, 89, 53, 67, 30, 73, 16, 69, 87, 40, 31,  0],
                            [ 4, 62, 98, 27, 23,  9, 70, 98, 73, 93, 38, 53, 60,  4, 23]], dtype=int)

# Build an even bigger matrix from a txt file
large_test_matrix = text_to_matrix('/Users/bwoodyear/desktop/p067_triangle.txt')


def max_path_finder(matrix):
    """
    Contains the parameters determining how the genetic algorithm works and runs it on a specified matrix. Prints the
    maximal path and graphs showing how the fitness changes during evolution.
    """
    size_population = 100
    best_sample = 30
    lucky_few = 10
    number_of_child = 5
    number_of_generation = 100
    chance_of_mutation = 70
    runs = 10

    temps1 = time.time()

    if (best_sample + lucky_few) / 2 * number_of_child != size_population:
        print("population size not stable")
    else:
        best_values = []
        for x in range(runs):
            historic = multiple_generations(number_of_generation, matrix, size_population, best_sample, lucky_few,
                                            number_of_child, chance_of_mutation)

            best_values.append(simple_result(historic, matrix, number_of_generation))
        print(max(best_values))

    evolution_best_fitness(historic, matrix)
    evolution_average_fitness(historic, matrix, size_population)

    print(time.time() - temps1)


max_path_finder(medium_test_matrix)
