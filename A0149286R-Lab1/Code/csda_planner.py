from dsda_planner import DSDAPlanner, main
import numpy as np
from math import sqrt, floor
from Queue import Queue

NUM_SPLIT_W = 4
NUM_SPLIT_V = 2

class CSDAPlanner(DSDAPlanner):
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3):
        super(CSDAPlanner, self).__init__(world_width, world_height, world_resolution, inflation_ratio)
        self.discretize_actions()
        # print(self.actions)
        self.holonomic_heuristic = {}

    def generate_plan(self):
        self.build_holonomic_heuristic()
        # print(self.holonomic_heuristic[(58,32)])
        # return
        super(CSDAPlanner, self).generate_plan()
        # self.action_seq = [(0, np.pi/2), (1,0), (1,0)]

    def discretize_actions(self):
        self.actions = []
        # v_split = int(round(NUM_SPLIT_V / 2))
        # for i in range(v_split + 1):
        for i in range(2):
            # v = float(i) / v_split
            # v = float(i) / 2
            v = i
            for j in range(NUM_SPLIT_W):
                w = np.pi * (-1 + 2.0 * j/NUM_SPLIT_W)
                self.actions.append((v, w))
        self.actions = tuple(self.actions)

    def discretize_state(self, state):
        x, y, theta = state
        theta = round(4 * theta/np.pi) % (2 * 4)
        return round(2 * NUM_SPLIT_V * x), round(2 * NUM_SPLIT_V * y), theta

    def get_current_state(self):
        return self.get_current_continuous_state()
    
    def get_motion_predict(self, x, y, theta, v, w):
        return self.motion_predict(x, y, theta, v, w)

    def heuristic(self, state):
        return max(self.euclidean_heuristic(state), self.get_holonomic_heuristic(state))

    def euclidean_heuristic(self, state):
        x, y, theta = state
        gx, gy = self._get_goal_position()
        dx = gx - x
        dy = gy - y
        return sqrt(dx * dx + dy * dy)

    def build_holonomic_heuristic(self):
        delta = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == j == 0: continue
                delta.append((i, j))
        
        x, y = self._get_goal_position()

        q = Queue()
        q.put((x, y))
        self.holonomic_heuristic[(2 * NUM_SPLIT_V * x, 2 * NUM_SPLIT_V * y)] = 0

        while not q.empty():
            cur_x, cur_y = q.get()
            # print(cur_x,cur_y)
            # if cur_x == cur_y == 1: break
            cur_stored_xy = (round(2 * NUM_SPLIT_V * cur_x), round(2 * NUM_SPLIT_V * cur_y))
            g = self.holonomic_heuristic[cur_stored_xy]
            for dx, dy in delta:
                nx = cur_x + dx / (2.0 * NUM_SPLIT_V)
                ny = cur_y + dy / (2.0 * NUM_SPLIT_V)
                stored_xy = (round(2 * NUM_SPLIT_V * nx), round(2 * NUM_SPLIT_V * ny))
                if stored_xy in self.holonomic_heuristic: continue
                if self.collision_checker(nx, ny): continue
                self.holonomic_heuristic[stored_xy] = g + 1 / (2.0 * NUM_SPLIT_V)
                q.put((nx, ny))

    def get_holonomic_heuristic(self, state):
        x, y, theta = state
        rounded = (round(2 * NUM_SPLIT_V * x), round(2 * NUM_SPLIT_V * y))
        if rounded not in self.holonomic_heuristic:
            return float("inf")
        return self.holonomic_heuristic[rounded]


def create_csda_planner(width, height, resolution, inflation_ratio, goal, publish=True):
    planner = CSDAPlanner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan()

    # You could replace this with other control publishers
    if publish:
        planner.publish_control()
    return planner

if __name__ == "__main__":
    main(create_csda_planner, 2)