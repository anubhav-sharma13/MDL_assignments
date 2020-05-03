import numpy
import random
import json
import requests
import client_moodle

population_size = 6 #no of parents 
number_of_weights = 11

def gen_mating_pool_probab(fitness) :
    probabilities = list(numpy.reciprocal(fitness) / float(sum(numpy.reciprocal(fitness))))
    return probabilities

def select_mates(probabs,genes):
    probabs.sort(reverse = True)
    if random.randint(0, 3) == 0:
        parent_index = numpy.random.choice(population_size, 1, probabs) #selects 2 parents based on the generated probabilities
        parents = [genes[parent_index[0]], genes[5]]
    else :
        parent_index = numpy.random.choice(population_size, 2, probabs) #selects 2 parents based on the generated probabilities
        parents = [genes[parent_index[0]], genes[parent_index[1]]]
    return parents

def crossover(parents): #crosses over two parent chromosomes
    child = []
    for i in range(len(parents[0])):
        x = random.randint(0,1)
        child.append(0.9*parents[x][i] + 0.1*parents[1-x][i])
    return child

def mutate(child):  #adds mutation to a child chromosome #abhi ismein sure nhi h....mtlb if my type doesn't work ..we have to look for other options as well
    for index in range(11):
        if random.randint(0,1):
            child[index] += 0.1*child[index]
        else:
            child[index] -= 0.1*child[index]
    return child

def find_mean_error(training_error, validation_error) : # #ensuring ki ham validation error ko bhi acchi khaasa weightage de rhe ho
    mean_err = (((training_error * validation_error * 1.3) / 10000000) + (training_error + validation_error / 9)) / 2
    return mean_err

for numb in range(10):

    with open('input.json','r') as f:
        parent_error = json.load(f)

    #parent_error is a list which contains lists of[[error,parent],[error,parent],[error,parent]]

    chromosomes = list(numpy.asarray(parent_error)[:,3])
    fitness = list(numpy.asarray(parent_error)[:,0])

    generations = 10
    child_number = 15
    min_training_error = parent_error[0][1]
    min_validation_error = parent_error[0][2]
    min_error = find_mean_error(min_training_error, min_validation_error)

    for generation in range(generations):

        child_errors = []
        chromosome_probability = gen_mating_pool_probab(fitness)

        for i in range(child_number):
            child = crossover(select_mates(chromosome_probability,chromosomes))
            if random.randint(0,3):
                mutate_child=child
            else:
                mutate_child = mutate(child)            
                
            err = client_moodle.get_errors('32EZBpTMqjBF5XByc7riOKvbIw2EykjhBEUqjDSAA9geHjqTaW', list(mutate_child))
            submit_status = client_moodle.submit('32EZBpTMqjBF5XByc7riOKvbIw2EykjhBEUqjDSAA9geHjqTaW', list(mutate_child))

            mean_error = find_mean_error(err[0], err[1])
            min_training_error = min(min_training_error, err[0])
            min_validation_error = min(min_validation_error, err[1])
            min_error = min(min_error, mean_error)
            
            child_errors.append([mean_error, err[0], err[1], child])
        
        random_values = numpy.array(random.sample(range(population_size), 5))

        for index in random_values:
            child_errors.append(parent_error[index])
        child_errors.sort()

        parent_error = child_errors[:population_size-1]
        parent_error.append(child_errors[population_size+2])
        fitness = list(numpy.asarray(parent_error)[:,0])
        
        print("Generation {} \n Population : {}".format(generation,parent_error))

    # #just sort over here once again if you like ...kyonki features sahi se aligned hone chahiye
    with open('input.json', 'w') as fjson:
        json.dump(parent_error[:population_size], fjson)