from base_planner import Planner, main
from heapq import heappush, heappop

class DSDAPlanner(Planner):
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3):
        super(DSDAPlanner, self).__init__(world_width, world_height, world_resolution, inflation_ratio)
        self.pq = []
        self.actions = ((1,0), (0,1), (0,-1))
        self.path_cost = {}

    def get_current_state(self):
        return self.get_current_discrete_state()
    
    def get_motion_predict(self, x, y, theta, v, w):
        return self.discrete_motion_predict(x, y, theta, v, w)
    
    def generate_plan(self):
        super(DSDAPlanner, self).generate_plan()
        # A* search
        s0 = self.get_current_state()
        # (f, state)
        heappush(self.pq, (self.heuristic(s0), s0))
        s0_d = self.discretize_state(s0)
        self.path_cost[s0_d] = 0

        i = 0

        while self.pq:
            f, cur_state = heappop(self.pq)
            x, y, theta = cur_state
            cur_state_d = self.discretize_state(cur_state)
            i += 1
            if i % 100 == 0:
                print(cur_state)
            if self._check_goal(cur_state):
                break

            for v, w in self.actions:
                # if cur_state_d == (1, 1, 1) and v == 1 and w == 0:
                #     print(theta)
                next_state = self.get_motion_predict(x, y, theta, v, w)
                if next_state is None: continue
                next_state_d = self.discretize_state(next_state)
                ng = self.path_cost[cur_state_d] + 1
                # if cur_state_d == (20,20, 0):
                #     print("???", next_state, next_state_d)
                if next_state_d in self.path_cost and self.path_cost[next_state_d] <= ng:
                    continue
                nf = ng + self.heuristic(next_state)
                self.path_cost[next_state_d] = ng
                self.action_table[next_state] = ((v, w), cur_state)

                heappush(self.pq, (nf, next_state))
        if not self._check_goal(cur_state):
            print("Goal not found")
            return
        self.get_path_to_state(cur_state)
    
    def discretize_state(self, state):
        return state

    def get_path_to_state(self, state):
        seq = []
        action, prev_state = self.action_table[state]
        seq.append(action)
        # print(state)
        while prev_state != self.initial_state:
            # print(prev_state)
            action, prev_state = self.action_table[prev_state]
            seq.append(action)
        self.action_seq = seq[::-1]
        # print(self.action_seq)


    def heuristic(self, state):
        # Manhattan Distance heuristic
        x, y, theta = state
        gx, gy = self._get_goal_position()
        return abs(x - gx) + abs(y - gy)

def create_dsda_planner(width, height, resolution, inflation_ratio, goal, publish=True):
    planner = DSDAPlanner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan()

    # You could replace this with other control publishers
    if publish:
        planner.publish_discrete_control()
    return planner

if __name__ == "__main__":
    main(create_dsda_planner)