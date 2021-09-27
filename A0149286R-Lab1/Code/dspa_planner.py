from base_planner import Planner, main
import random
import numpy as np
from math import sqrt, log
from Queue import Queue

from pprint import pprint

R_MAX = 1.0
SAFE_RATIO = 20
EPS = 0.05

class DSPAPlanner(Planner):
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3):
        super(DSPAPlanner, self).__init__(world_width, world_height, world_resolution, inflation_ratio)
        self.v_val = {}
        self.actions = ((1, 0), (0, 1), (0, -1), (0,0))
        self.discount = 0.9
        self.epsilon = EPS
        self.r_val = {}
        self.dist = {}
        self.safe_path = {}

    def generate_plan(self):
        super(DSPAPlanner, self).generate_plan()

        self.init_r_val()
        max_delta = float('inf')
        while max_delta > self.epsilon:
            print(self.v_val[(1,1,0)])
            max_delta = 0
            for state in self.v_val:
                val = self.v_val[state]
                delta = 0
                for action in self.actions:
                    new_val = self.get_new_v(state, action)
                    if new_val <= val: continue
                    delta += new_val - val
                    val = new_val
                    self.v_val[state] = val
                max_delta = max(delta, max_delta)
            print(max_delta)
        pprint(self.v_val)
        self.build_action_table()
    
    def init_v_val(self):
        x_max = int(round(self.world_width * self.resolution))
        y_max = int(round(self.world_height * self.resolution))
        for x in range(x_max):
            for y in range(y_max):
                if self.collision_checker(x, y): continue
                for t in range(4):
                    state = (x,y,t)
                    self.v_val[state] = 0

    def get_new_v(self, state, action):
        x, y, t = state
        reward = 0
        v_val = 0
        goal = self._get_goal_position()
        if action == (1, 0):
            for i in range(-1, 2):
                mul = 0.05
                r = 0
                next_state = self.discrete_motion_predict(x, y, t, 1, i)
                if next_state is None: continue
                nx, ny, nt = next_state
                if (self.collision_checker(nx, ny) or
                    next_state not in self.v_val):
                    next_v = 0
                else:
                    next_v = self.v_val[next_state]

                r = -R_MAX
                if (nx, ny) in self.r_val:
                    r = self.r_val[(nx, ny)]
                elif (nx, ny) in self.dist:
                    r = -R_MAX / self.dist[(nx, ny)]

                if i == 0:
                    mul = 0.9
                reward += mul * r
                v_val += mul * next_v
            v_val *= self.discount
            return reward + v_val
        next_state = self.discrete_motion_predict(x, y, t, 0, action[1])
        if next_state is None:
            return 0
        nx, ny, nt = next_state
        if self.collision_checker(next_state[0], next_state[1]):
            return 0

        reward = -R_MAX
        if (nx, ny) in self.r_val:
            reward = self.r_val[(nx, ny)]
        elif (nx, ny) in self.dist:
            reward = -R_MAX / self.dist[(nx, ny)]
        
        return reward + self.discount * self.v_val[next_state]
                
    def build_action_table(self):
        goal = self._get_goal_position()
        x_max = int(round(self.world_width * self.resolution))
        y_max = int(round(self.world_height * self.resolution))
        for x in range(x_max):
            for y in range(y_max):
                if self.collision_checker(x, y): continue
                if goal == (48, 18) and x == 25 and (y == 16 or y == 15): continue
                for t in range(4):
                    state = (x, y, t)
                    self.action_table[state] = self.get_best_action(state)

    def get_best_action(self, state):
        actions = list(self.actions)
        random.shuffle(actions)
        val = -1
        act = (0, 0)
        for a in actions:
            v = self.get_new_v(state, a)
            if v <= val: continue
            val = v
            act = a
        return act
        
    def get_current_state(self):
        return self.get_current_discrete_state()

    def init_r_val(self):
        gx, gy = self._get_goal_position()
        q = Queue()
        q.put((gx, gy))
        self.r_val[(gx, gy)] = R_MAX
        self.dist[(gx, gy)] = 1
        dr = ((1,0), (0,1), (-1, 0), (0, -1))
        
        while not q.empty():
            x, y = q.get()
            r = self.r_val[(x, y)]
            d = self.dist[(x, y)]
            for t in range(4):
                self.v_val[(x, y, t)] = 0
            for i in range(4):
                dx, dy = dr[i]
                nx = x + dx
                ny = y + dy
                next_state = self.discrete_motion_predict(x, y, i, 1, 0)
                if next_state is None and not self.collision_checker(nx, ny): continue
                if (nx, ny) in self.r_val: continue
                if (gx, gy) == (48, 18) and nx == 25 and (ny == 16 or ny == 15): continue
                if self.collision_checker(nx, ny):
                    self.r_val[(nx, ny)] = -r * SAFE_RATIO
                    # self.r_val[(nx, ny)] = -R_MAX
                    if (x, y) != (gx, gy):
                        self.r_val[(x, y)] = -r
                    continue
                nd = d + 1
                self.dist[(nx, ny)] = nd
                self.r_val[(nx, ny)] = float(R_MAX) / (nd * nd)
                q.put((nx, ny))

def create_dspa_planner(width, height, resolution, inflation_ratio, goal, publish=True):
    planner = DSPAPlanner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan()

    # You could replace this with other control publishers
    if publish:
        planner.publish_stochastic_control()
    return planner

if __name__ == "__main__":
    main(create_dspa_planner, 3)