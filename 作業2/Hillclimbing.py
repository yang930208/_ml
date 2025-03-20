import random
import math
from typing import List, Tuple


class TravelingSalesmanProblem:
    def __init__(self, cities: List[Tuple[int, int]]):
        self.cities = cities
        self.num_cities = len(cities)

    def distance(self, city1: Tuple[int, int], city2: Tuple[int, int]) -> float:
        return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

    def total_distance(self, path: List[int]) -> float:
        dist = 0.0
        for i in range(self.num_cities):
            dist += self.distance(self.cities[path[i]], self.cities[path[(i + 1) % self.num_cities]])
        return dist

    def random_solution(self) -> List[int]:
        path = list(range(self.num_cities))
        random.shuffle(path)
        return path

    def get_neighbors(self, path: List[int]) -> List[List[int]]:
        neighbors = []
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                new_path = path[:]
                new_path[i], new_path[j] = new_path[j], new_path[i]
                neighbors.append(new_path)
        return neighbors

    def hill_climbing(self, max_iterations: int = 1000) -> Tuple[List[int], float]:
        current_solution = self.random_solution()
        current_cost = self.total_distance(current_solution)

        for _ in range(max_iterations):
            neighbors = self.get_neighbors(current_solution)
            best_neighbor = current_solution
            best_cost = current_cost

            for neighbor in neighbors:
                neighbor_cost = self.total_distance(neighbor)
                if neighbor_cost < best_cost:  # 如果找到更好的解
                    best_neighbor = neighbor
                    best_cost = neighbor_cost

            if best_cost >= current_cost:  # 無法改善，停止搜尋
                break

            current_solution, current_cost = best_neighbor, best_cost

        return current_solution, current_cost


if __name__ == "__main__":
    # 隨機生成 10 個城市的位置
    cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
    tsp = TravelingSalesmanProblem(cities)

    # 執行爬山演算法
    best_path, best_cost = tsp.hill_climbing()

    print(f"最佳路徑：{best_path}")
    print(f"最短距離：{best_cost}")
# 採用chatgpt生成 https://chatgpt.com/canvas/shared/67db817aebbc8191813c4a8eeca1e6b3