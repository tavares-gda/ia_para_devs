import random
import pandas as pd


class CaixeiroViajanteICMS:
    def __init__(self, capitals, distances, icms_rates, connections, valor_bem=1000000, custo_por_km=1,
                 population_size=100, generations=500, mutation_rate=0.1):
        self.capitals = capitals
        self.distances = distances
        self.icms_rates = icms_rates
        self.connections = connections
        self.valor_bem = valor_bem
        self.custo_por_km = custo_por_km
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self, origin, destination):
        possible_routes = []
        max_attempts = 1000

        def dfs_route(current_city, visited):
            if current_city == destination:
                return [destination]
            neighbors = self.connections.get(current_city, [])
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path = dfs_route(neighbor, visited)
                    if path:
                        return [current_city] + path
                    visited.remove(neighbor)
            return None

        attempts = 0
        while len(possible_routes) < self.population_size and attempts < max_attempts:
            route = dfs_route(origin, set([origin]))
            if route and route[-1] == destination:
                possible_routes.append(route)
            attempts += 1

        if len(possible_routes) < self.population_size:
            raise ValueError("Não foi possível gerar rotas válidas suficientes respeitando as conexões diretas.")
        self.population = possible_routes

    def calculate_fitness(self, route):
        total_distance_cost = 0
        total_icms_cost = 0
        for i in range(len(route) - 1):
            start, end = route[i], route[i + 1]
            if end not in self.connections.get(start, []):
                return float('inf')

            distance = self.distances[self.capitals.index(start)][self.capitals.index(end)]
            total_distance_cost += distance * self.custo_por_km
            total_icms_cost += self.icms_rates[start] * self.valor_bem

        fitness = total_distance_cost + total_icms_cost
        return fitness

    def sort_population_by_fitness(self):
        self.population.sort(key=self.calculate_fitness)

    def crossover(self, parent1, parent2, destination):
        if len(parent1) <= 2 or len(parent2) <= 2:
            return parent1, parent2

        cut = random.randint(1, min(len(parent1), len(parent2)) - 2)
        child1, child2 = parent1[:cut], parent2[:cut]

        def extend_route(route, remaining_route):
            for city in remaining_route:
                if city in self.connections.get(route[-1], []) and city != destination:
                    route.append(city)
            if route[-1] != destination and destination in self.connections.get(route[-1], []):
                route.append(destination)
            return route

        child1 = extend_route(child1, parent2[cut:])
        child2 = extend_route(child2, parent1[cut:])
        return child1, child2

    def mutate(self, route, destination):
        if random.random() < self.mutation_rate and len(route) > 3:
            if len(route) - 2 >= 2:
                i, j = sorted(random.sample(range(1, len(route) - 1), 2))
                if (route[j] in self.connections.get(route[i - 1], []) and
                        route[j + 1] in self.connections.get(route[j], [])):
                    route[i], route[j] = route[j], route[i]
        if route[-1] != destination:
            route[-1] = destination
        return route

    def evolve_population(self, destination):
        new_population = self.population[:2]  # Elitism
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(self.population[:10], 2)
            child1, child2 = self.crossover(parent1, parent2, destination)
            new_population.append(self.mutate(child1, destination))
            new_population.append(self.mutate(child2, destination))
        self.population = new_population

    def run(self, origin, destination):
        self.initialize_population(origin, destination)
        for gen in range(self.generations):
            self.sort_population_by_fitness()
            current_best_fitness = self.calculate_fitness(self.population[0])
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[0]
            self.evolve_population(destination)

        return self.best_solution, self.best_fitness


connections_df = pd.read_excel('caminhos_possiveis.xlsx', sheet_name='Sheet1',
                               index_col=0)
connections = {row.Index: row.Conexoes.split(',') for row in connections_df.itertuples()}
distances_df = pd.read_excel('distancias_capitais.xlsx', index_col=0)
distances = distances_df.values
capitals = distances_df.columns.tolist()

icms_rates = {
    "SP": 0.18, "RJ": 0.22, "MG": 0.18, "RS": 0.17, "PR": 0.195, "SC": 0.17, "BA": 0.205, "DF": 0.20,
    "GO": 0.19, "PA": 0.19, "MT": 0.17, "PE": 0.205, "CE": 0.20, "ES": 0.17, "MS": 0.17, "AM": 0.20,
    "MA": 0.22, "RN": 0.18, "PB": 0.20, "AL": 0.19, "PI": 0.21, "RO": 0.195, "SE": 0.19, "TO": 0.20,
    "AC": 0.19, "AP": 0.18, "RR": 0.20
}

for origin, destination in [("AM", "SP"), ("CE", "RS"), ("MT", "PI"), ("SP", "AM"), ("RS", "PE"), ("AM", "GO")]:
    instance = CaixeiroViajanteICMS(capitals, distances, icms_rates, connections, valor_bem=1000000, custo_por_km=1,
                                    population_size=50, generations=100, mutation_rate=0.2)
    best_route, best_cost = instance.run(origin, destination)
    print(f"Best Route from {origin} to {destination}:", best_route)
    print(f"Best Cost (Distance + ICMS):", best_cost)
