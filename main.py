import pandas as pd
import random
import pygad
import numpy


def get_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    return train, test


def test(ga_instance, solution, solution_idx):
    fitness = 0
    for index, row in test_data.iterrows():
        output = numpy.sum(solution[1:] * row[1:11]) + solution[0]
        fitness += (output - row[11]) ** 2
    return fitness


def fitness_func(ga_instance, solution, solution_idx):
    fitness = 0
    for index, row in train_data.iterrows():
        output = numpy.sum(solution[1:] * row[1:11]) + solution[0]
        fitness += (output - row[11]) ** 2
    return 1.0 / fitness


if __name__ == "__main__":
    train_data, test_data = get_data()

    fitness_function = fitness_func

    ga_instance = pygad.GA(num_generations=50,
                           num_parents_mating=4,
                           fitness_func=fitness_function,
                           sol_per_pop=8,
                           num_genes=11,
                           parent_selection_type="sss",
                           keep_parents=1,
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=10)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Parameters of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    print(test(ga_instance, solution, solution_idx))

