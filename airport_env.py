import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class AirportEnv(gym.Env):
    def __init__(self):
        super(AirportEnv, self).__init__()
        
        self.TOTAL_EPISODE_STEPS = 300       # 1 step = 1 min, 300 mins = 5 hr 
        self.TOTAL_PAX_UPPERLIMIT = 37752
        self.TOTAL_PAX_LOWERLIMIT = 25168
        self.CHECKIN_SERVICE_TIME = 2        # each pax requires 2 steps. This service is computed differently from the other two
        self.SECURITY_SERVICE_RATE = 3       # 3 pax per step per lane.
        self.BOARDING_SERVICE_RATE = 9       # 9 pax per step per gate.

        self.TOTAL_CHECKIN_COUNTER = 192
        self.TOTAL_SECURITY_LANE = 53
        self.TOTAL_BOARDING_GATE = 60

        self.BUFFER_MAX_TIME_CHECKIN_TO_SECURITY = 3
        self.BUFFER_MAX_TIME_SECURITY_TO_BOARDING = 2

        self.REWARD = 0
        self.alpha = 0.05
        self.beta = 0.5
        self.gamma = 0.2
        self.service_switch_time = 5

        # Selecting Staff randomly before start of episode
        self.TOTAL_SERVICE_UPPER_LIMIT = 290  # roughly 95% of total staff required
        self.TOTAL_SERVICE_LOWER_LIMIT = 275  # roughly 90% of total staff required
        

        # Model Actions
        self.action_space = gym.spaces.Discrete(7)
        # Actions:
        # 0: Do nothing
        # 1: Move service from check-in → security
        # 2: Move service from check-in → boarding
        # 3: Move service from security → check-in
        # 4: Move service from security → boarding
        # 5: Move service from boarding → check-in
        # 6: Move service from boarding → security


        # Observation Space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.TOTAL_EPISODE_STEPS, self.TOTAL_CHECKIN_COUNTER, self.TOTAL_SECURITY_LANE, self.TOTAL_BOARDING_GATE, self.service_switch_time, self.TOTAL_PAX_UPPERLIMIT, self.TOTAL_PAX_UPPERLIMIT, self.TOTAL_PAX_UPPERLIMIT, self.TOTAL_PAX_UPPERLIMIT, self.TOTAL_PAX_UPPERLIMIT, self.TOTAL_PAX_UPPERLIMIT], dtype=np.float32),
            shape=(11,),
            dtype=np.float32
        )

        # obs = np.array([
        #     self.step_count,
        #     self.checkin_counter,
        #     self.security_lanes,
        #     self.boarding_gates,
        #     len(self.service_switch_queue),
        #     len(self.wait_checkin_queue),
        #     self.wait_security_queue_len,
        #     self.wait_boarding_queue_len,
        #     len(self.buffer_checkin_to_security),
        #     len(self.buffer_security_to_boarding),
        #     self.reward_R1
        # ], dtype=np.float32)


    def reset(self, seed=None, options=None):
        # --- Set the random seed for reproducibility ---
        super().reset(seed=seed)  # Gymnasium's internal call
        np.random.seed(seed)
        random.seed(seed)
        
        self.step_count = 0
        self.total_pax_in_episode = np.random.randint(self.TOTAL_PAX_LOWERLIMIT, self.TOTAL_PAX_UPPERLIMIT + 1)
        self.starting_service = np.random.randint(self.TOTAL_SERVICE_LOWER_LIMIT, self.TOTAL_SERVICE_UPPER_LIMIT + 1)

        self.reward_R1 = 0  # positive reward = pax processed at each step

        # Staff Allocation
        self.checkin_counters = round(0.63 * self.starting_service)
        self.security_lanes = round(0.17 * self.starting_service)
        self.boarding_gates = round(0.19 * self.starting_service)

        # Negative reward = operational staff cost
        self.reward_R2 = self.checkin_counters + self.security_lanes + self.boarding_gates

        # Initialize queues
        self.wait_checkin_queue = []
        self.wait_security_queue_len = 0
        self.wait_boarding_queue_len = 0
        self.service_switch_queue = []
        self.buffer_checkin_to_security = []
        self.buffer_security_to_boarding = []

        # Pax Arrival
        self.step_pax_arrivals = np.random.poisson(lam=self.total_pax_in_episode / 300, size=300)

        obs = np.array([
            self.step_count,
            self.checkin_counters,
            self.security_lanes,
            self.boarding_gates,
            len(self.service_switch_queue),
            len(self.wait_checkin_queue),
            self.wait_security_queue_len,
            self.wait_boarding_queue_len,
            len(self.buffer_checkin_to_security),
            len(self.buffer_security_to_boarding),
            self.reward_R1
        ], dtype=np.float32)

        return obs, {}


    def step(self, action=0):

        # Adding Pax from new step to checkin queue
        for i in range (0, self.step_pax_arrivals[self.step_count]):
            self.wait_checkin_queue.append(self.CHECKIN_SERVICE_TIME)

        self.step_count+=1

        # Processing Queues in reverse order. Pax must get <= one service at one step.

        # Processing Service Switch Queue
        k=0
        for i in range (0, len(self.service_switch_queue)):
            self.service_switch_queue[k] = (self.service_switch_queue[k][0], self.service_switch_queue[k][1]-1)
            if self.service_switch_queue[k][1]==0:
                if self.service_switch_queue[k][0]=='To_Checkin':
                    self.checkin_counters += 1
                elif self.service_switch_queue[k][0]=='To_Security':
                    self.security_lanes += 1
                else:
                    self.boarding_gates += 1
                self.service_switch_queue.pop(k)
            else:
                k+=1

        # Processing Boarding Queue
        pax_processed = min(self.BOARDING_SERVICE_RATE*self.boarding_gates, self.wait_boarding_queue_len)
        self.wait_boarding_queue_len -= pax_processed

        # Processing Buffer Security to Boarding
        loop_itr = len(self.buffer_security_to_boarding) 
        k=0
        for i in range (0, loop_itr):
            self.buffer_security_to_boarding[k]-=1
            if (self.buffer_security_to_boarding[k]==0):
                self.wait_boarding_queue_len += 1
                self.buffer_security_to_boarding.pop(k)
            else:
                k+=1

        # Processing Security Queues
        security_processed = min(self.SECURITY_SERVICE_RATE*self.security_lanes, self.wait_security_queue_len)
        self.wait_security_queue_len -= security_processed

        for i in range (0, security_processed):
            self.buffer_security_to_boarding.append(random.randint(1, self.BUFFER_MAX_TIME_SECURITY_TO_BOARDING))
        
        # Processing Buffer Checkin to Security
        loop_itr = len(self.buffer_checkin_to_security)
        k = 0
        for i in range (0, loop_itr):
            self.buffer_checkin_to_security[k]-=1
            if (self.buffer_checkin_to_security[k]==0):
                self.wait_security_queue_len+=1
                self.buffer_checkin_to_security.pop(k)
            else:
                k+=1

        # Processing Checkin Queue
        loop_itr = min(len(self.wait_checkin_queue), self.checkin_counters)
        k = 0
        for i in range(0, loop_itr):
            self.wait_checkin_queue[k]-=1
            if (self.wait_checkin_queue[k]==0):
                self.buffer_checkin_to_security.append(random.randint(1, self.BUFFER_MAX_TIME_CHECKIN_TO_SECURITY))
                self.wait_checkin_queue.pop(k)
            
            else:
                k+=1

        illegal_action_penalty = 0
        # Implement Action (0 = do Nothing, 1-6 = Move 3 staff across service)
        
        if action==1:
            # Move service checkin to security
            if self.security_lanes == self.TOTAL_SECURITY_LANE or self.checkin_counters==0:
                # Action should be heavily penalizaed
                illegal_action_penalty = 1

            else:
                self.checkin_counters -= 1
                self.service_switch_queue.append(("To_Security", self.service_switch_time))

        elif action==2:
            # Move 3 staff from checkin to boarding
            if self.boarding_gates == self.TOTAL_BOARDING_GATE or self.checkin_counters==0:
                # Action should be heavily penalizaed
                illegal_action_penalty = 1

            else:
                self.checkin_counters -= 1
                self.service_switch_queue.append(("To_Boarding", self.service_switch_time))
                

        elif action==3:
            # Move 3 staff from security to checkin
            if self.checkin_counters == self.TOTAL_CHECKIN_COUNTER or self.security_lanes==0:
                # Action should be heavily penalizaed
                illegal_action_penalty = 1

            else:
                self.security_lanes -= 1
                self.service_switch_queue.append(("To_Checkin", self.service_switch_time))

        elif action==4:
            # Move 3 staff from security to boarding
            if self.boarding_gates == self.TOTAL_BOARDING_GATE or self.security_lanes==0:
                # Action should be heavily penalizaed
                illegal_action_penalty = 1

            else:
                self.security_lanes -= 1
                self.service_switch_queue.append(("To_Boarding", self.service_switch_time))

        elif action==5:
            # Move 3 staff from boarding to checkin
            if self.checkin_counters == self.TOTAL_CHECKIN_COUNTER or self.boarding_gates==0:
                # Action should be heavily penalizaed
                illegal_action_penalty = 1

            else:
                self.boarding_gates -= 1
                self.service_switch_queue.append(("To_Checkin", self.service_switch_time))

        elif action==6:
            # Move 3 staff from boarding to security
            if self.security_lanes == self.TOTAL_SECURITY_LANE or self.boarding_gates==0:
                # Action should be heavily penalizaed
                illegal_action_penalty = 1

            else:
                self.boarding_gates -= 1
                self.service_switch_queue.append(("To_Security", self.service_switch_time))

        # Reward Computation for state:
        self.reward_R1 = pax_processed
        
        reward = (self.reward_R1/(self.TOTAL_BOARDING_GATE*self.BOARDING_SERVICE_RATE)) - self.alpha * (action != 0) - self.beta * (len(self.wait_checkin_queue) + self.wait_boarding_queue_len + self.wait_security_queue_len)/(self.TOTAL_PAX_UPPERLIMIT) - self.gamma*illegal_action_penalty
        self.REWARD += reward
        
    
        obs = np.array([
            self.step_count,
            self.checkin_counters,
            self.security_lanes,
            self.boarding_gates,
            len(self.service_switch_queue),
            len(self.wait_checkin_queue),
            self.wait_security_queue_len,
            self.wait_boarding_queue_len,
            len(self.buffer_checkin_to_security),
            len(self.buffer_security_to_boarding),
            self.reward_R1
        ], dtype=np.float32)

        done = self.step_count >= self.TOTAL_EPISODE_STEPS
        return obs, reward, done, False, {}
    
    def render(self, mode='human'):
        """
        Render the current simulation state for debugging and visualization.
        This function prints a structured summary of the environment at each step.
        """
        print("\n" + "="*60)
        print(f" STEP {self.step_count}")
        print("="*60)

        # --- Passenger Info ---
        print(" Passenger Flow")
        print(f"  Arrivals this step       : {self.step_pax_arrivals[self.step_count-1] if self.step_count > 0 else 0}")
        print(f"  Total passengers in episode : {self.total_pax_in_episode if self.step_count > 0 else 0}")

        # --- Service Distribution ---
        print("\n Service Allocation")
        print(f"  Check-in counters  : {self.checkin_counters}")
        print(f"  Security Lanes  : {self.security_lanes}")
        print(f"  Boarding Gates  : {self.boarding_gates}")

        # --- Queue & Buffer Lengths ---
        print("\n Queue and Buffer Status")
        print(f"  Check-in Queue Length         : {len(self.wait_checkin_queue)}")
        print(f"  Security Queue Length         : {self.wait_security_queue_len}")
        print(f"  Boarding Queue Length         : {self.wait_boarding_queue_len}")
        print(f"  Buffer Check-in ➜ Security    : {len(self.buffer_checkin_to_security)}")
        print(f"  Buffer Security ➜ Boarding    : {len(self.buffer_security_to_boarding)}")
        print(f"  Buffer for Service Switch      : {len(self.service_switch_queue)}")

        # --- Rewards ---
        print("\n Rewards")
        print(f"  R1 (Processed Pax)  : {self.reward_R1:.2f}")
        print(f"  R2 (Staff Cost)  : {self.reward_R2:.2f}")
        print(f"  Total Reward so far : {self.REWARD:.2f}")

        print("="*60)
