import random
import time

class GeneticAlgorithmSolver:
    """
    A genetic algorithm solver for the Traveling Salesman Problem.
    Meta parameters are randomly generated during initialization.
    """

    def set_meta_config( self, population_size, cull_rate, mutation_rate, crossover_rate, num_generations ):
        self.population_size = population_size
        self.cull_rate = cull_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations


    def __init__(self):
        # Randomly generate meta parameters for the genetic algorithm.
        self.population_size = random.randint(50, 150)       # e.g., between 50 and 150 individuals
        self.cull_rate = random.uniform(0.5, 0.9)            # e.g., between 50% and 90%
        self.mutation_rate = random.uniform(0.01, 0.1)       # e.g., between 1% and 10%
        self.crossover_rate = random.uniform(0.5, 0.9)       # e.g., between 50% and 90%
        self.num_generations = random.randint(100, 500)      # e.g., between 100 and 500 generations

    def solve(self, graph, timestart, timelimit=None):
        """
        Solves the TSP using a genetic algorithm.

        Parameters:
            graph: A TSP instance (e.g. loaded using tsplib95) that provides:
                   - get_nodes() returning the set/list of nodes.
                   - get_weight(i, j) returning the distance between nodes i and j.

        Returns:
            A dictionary with:
              - "tour": A list representing the order in which nodes are visited,
                        starting and ending at the same node.
              - "cost": The total cost (distance) of the computed tour.
        """

        # Just to keep things finishing in a reasonable timeline
        # we'll kill all jobs that take longer than 10 mins (Since they'd take too long
        # in a realworld applications)
        if not timelimit:
            timelimit = 10 * 60

        nodes = list(graph.get_nodes())
        if not nodes:
            return {"tour": [], "cost": 0}

        # Helper function to compute the total cost of a tour.
        def compute_cost(tour):
            return sum(graph.get_weight(tour[i], tour[i + 1]) for i in range(len(tour) - 1))
        
        # Create an individual (a candidate tour).
        # We fix the starting node to be the first in the list.
        def create_individual():
            individual = nodes[1:]  # exclude the first node
            random.shuffle(individual)
            # Build a tour: start -> permutation -> start
            return [nodes[0]] + individual + [nodes[0]]
        
        # Initialize the population.
        population = [create_individual() for _ in range(self.population_size)]
        
        # crossover (OX) for TSP.
        def crossover(parent1, parent2):
            n = len(parent1)
            # Choose a single crossover point (between 1 and n-2 to keep endpoints fixed)
            cp = random.randint(1, n - 1)

            # Initialize the child with parent 1 up to cross over point
            child = parent1[:cp]

            # we can use cp as a way to trakc 

            # build the child using p2 as reference, but only if they are available!
            while len(child) < n - 1:
                # try to pull from parent 2 first
                candidate = parent2[cp]

                if candidate in child:
                    # if not possible, then select any random node that is available
                    missing = [node for node in parent1 if node not in child]
                    candidate = random.choice(missing)

                child.append(candidate)
                cp += 1 # bump to the next!

            child.append(parent1[0])

            return child
        
        # Swap mutation: swap two random nodes (except the fixed first/last node).
        def mutate(individual):
            if random.random() < self.mutation_rate:
                i, j = random.sample(range(1, len(individual) - 1), 2)
                individual[i], individual[j] = individual[j], individual[i]
        
        best_individual = None
        best_cost = float('inf')

        # Main GA loop, generating the next generation
        for generation in range(self.num_generations):
            # Enforce Time Limit
            if timelimit and (timelimit < time.time() - timestart):
                break;

            # Sort current population based on tour cost.
            population.sort(key=lambda ind: compute_cost(ind))

            if best_cost > compute_cost( population[0] ):
                best_individual = population[0]
                best_cost = compute_cost( best_individual )

            new_population = []

            # Elitism: carry over the best individuals (50 percent cull)
            culled_pop_size = int( len(population) * self.cull_rate )
            new_population = population[:culled_pop_size]
            
            # Generate new individuals until the new population is full.
            while len(new_population) < self.population_size:
                parent1 = random.choice(population)
                parent2 = random.choice(population)

                # default to parent1
                child = parent1

                # Apply Genetic Operations (only 1!)
                if random.random() < self.crossover_rate:
                    child = crossover(parent1, parent2)
                elif random.random() < self.mutation_rate:
                    mutate(child)
                new_population.append(child)

            population = new_population

        # Sort current population based on tour cost
        population.sort(key=lambda ind: compute_cost(ind))

        if best_cost > compute_cost( population[0] ):
            best_individual = population[0]
            best_cost = compute_cost( best_individual )
       
        final_cost = compute_cost( best_individual )

        return {"tour": best_individual, "cost": final_cost, "meta": vars(self)}
