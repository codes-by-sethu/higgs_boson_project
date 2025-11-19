import numpy as np
import random
from deap import base, creator, tools, algorithms, gp
from sklearn.metrics import accuracy_score
import operator

class EvolutionaryLearner:
    def __init__(self, n_features, primitive_set=None, population_size=100, 
                 generations=50, cx_prob=0.5, mut_prob=0.2):
        self.n_features = n_features
        self.population_size = population_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        
        # Initialize DEAP
        self._setup_deap()
        
        # Primitive set
        self.pset = self._create_primitive_set(primitive_set)
        
        # Statistics
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
    
    def _setup_deap(self):
        """Setup DEAP creator and toolbox"""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
    def _create_primitive_set(self, custom_pset):
        """Create primitive set for genetic programming"""
        pset = gp.PrimitiveSet("MAIN", self.n_features)
        
        # Arithmetic operations
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(operator.neg, 1)
        pset.addPrimitive(self.protected_div, 2)
        
        # Mathematical functions
        pset.addPrimitive(np.sin, 1)
        pset.addPrimitive(np.cos, 1)
        pset.addPrimitive(np.tanh, 1)
        
        # Terminals
        pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
        
        # Add custom primitives if provided
        if custom_pset:
            for prim in custom_pset:
                pset.addPrimitive(prim[0], prim[1])
        
        return pset
    
    def protected_div(self, left, right):
        """Protected division to avoid division by zero"""
        try:
            return left / right if right != 0 else 1
        except:
            return 1
    
    def eval_classifier(self, individual, X, y):
        """Evaluate individual as a classifier"""
        try:
            # Compile the individual into a function
            func = self.toolbox.compile(expr=individual)
            
            # Apply function to all samples
            predictions = []
            for sample in X:
                try:
                    output = func(*sample)
                    # Convert to binary classification
                    pred = 1 if output > 0 else 0
                    predictions.append(pred)
                except:
                    predictions.append(0)  # Default prediction
                    
            accuracy = accuracy_score(y, predictions)
            return accuracy,
            
        except:
            return 0.0,  # Return minimum fitness
    
    def initialize_evolution(self, X_train, y_train):
        """Initialize evolutionary learning process"""
        # Register genetic operations
        self.toolbox.register("expr", gp.genHalfAndHalf, 
                             pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, 
                             creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, 
                             list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        # Register evolutionary operators
        self.toolbox.register("evaluate", self.eval_classifier, 
                             X=X_train, y=y_train)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, 
                             expr=self.toolbox.expr_mut, pset=self.pset)
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        return population
    
    def run_evolution(self, X_train, y_train):
        """Run the evolutionary learning process"""
        population = self.initialize_evolution(X_train, y_train)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + self.stats.fields
        
        # Evolutionary loop
        for gen in range(self.generations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Record statistics
            record = self.stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {record['max']:.4f}")
        
        return population, logbook
    
    def predict(self, individual, X):
        """Make predictions using an individual"""
        func = self.toolbox.compile(expr=individual)
        predictions = []
        for sample in X:
            try:
                output = func(*sample)
                pred = 1 if output > 0 else 0
                predictions.append(pred)
            except:
                predictions.append(0)
        return np.array(predictions)

if __name__ == "__main__":
    print("Evolutionary Learning module created successfully.")