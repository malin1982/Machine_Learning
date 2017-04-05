import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qvals = {}
        self.possible_actions = (None, 'forward', 'left', 'right')
        self.total_reward = 0
        self.moves = 1
        self.penalties = 0
        self.fail = 0
        self.next_state = None
        self.epsilon = 0.05
    def update_qvals(self, state, action, reward, next_state):
        learn_rate = 1.0/math.log(self.moves+5)
        discount_factor = 0.1
        #print learn_rate
        #init qvals is 1
        #print self.state
        #print self.next_state
        #print "current q value is: {}".format(self.qvals.get((self.state, action),0))
        #print {action: self.qvals.get((self.state,action), 0) for action in self.possible_actions}.values()

        self.qvals[(self.state, action)] = (1- learn_rate) * self.qvals.get((self.state, action), 0) \
                                                         + learn_rate * (reward \
                                                            + discount_factor * max({action: self.qvals.get((self.next_state,action), 0) for action in self.possible_actions}.values()))
        #print "next max q value is: {}".format(next_max_qval)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def get_best_action(self, state):
        all_qvals = {action: self.qvals.get((self.state, action), 0) for action in self.possible_actions}
        best_action = [action for action in self.possible_actions if all_qvals[action] == max(all_qvals.values())]
        return random.choice(best_action)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state

        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)

        # self.state = (inputs['light'], self.next_waypoint, \
        #     ( inputs['light']== 'green' and inputs['oncoming'] == 'forward'), \
        #     (inputs['light'] == 'red' and inputs['left'] == 'forward'))
        #print self.state
        #print all_qvals
        if random.random() < self.epsilon:
            action = random.choice(Environment.valid_actions[1:])
            reward = self.env.act(self, action)
            next_inputs = self.env.sense(self)
            self.next_waypoint = self.planner.next_waypoint()
            self.next_state = (next_inputs['light'], next_inputs['oncoming'], next_inputs['left'], next_inputs['right'], self.next_waypoint)
            self.update_qvals(self.state, action, reward, self.next_state)
        else:
            action = self.get_best_action(self.state)
            reward = self.env.act(self, action)
            next_inputs = self.env.sense(self)
            self.next_waypoint = self.planner.next_waypoint()
            self.next_state = (next_inputs['light'], next_inputs['oncoming'], next_inputs['left'], next_inputs['right'], self.next_waypoint)
            self.update_qvals(self.state, action, reward, self.next_state)


        # TODO: Select action according to your policy
        #action = random.choice(Environment.valid_actions[1:])

        # Execute action and get reward
        #reward = self.env.act(self, action)
        # TODO: Learn policy based on state, action, reward
        self.total_reward += reward
        self.moves += 1
        if reward < 0:
            self.penalties+= 1
        if deadline <= 0:
            self.fail+=1
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    #print a.qvals
    print "total reward is:{}; total moves is:{}; total penalties is:{}; penalty rate is:{}; total fail is:{}".format(a.total_reward, a.moves, a.penalties, float(a.penalties)/a.moves, a.fail)

if __name__ == '__main__':
    run()
