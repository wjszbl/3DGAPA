class Individual:
    def __init__(self, assignment=None):
        if assignment is None:
            self.assignment = [random.choice([0, 1]) for _ in range(n_items)]
        else:
            self.assignment = assignment
        self.fitness = self.calculate_fitness()

    @staticmethod
    def calculate_fitness(assignment):
        return sum(assignment)
