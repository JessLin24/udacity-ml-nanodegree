import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3 # For each timestep of the agent, we step the simulation action_repeats timesteps

        self.state_size = self.action_repeat * 6 # action repeats * 6-dimensional pose info (for current sample)
        self.action_low = 0 # minimum value of entry (revolutions/sec of rotor)
        self.action_high = 900 # maximum value of entry (revolutions/sec of rotor)
        self.action_size = 4 # Environment wil always have 4-dimensional action space; one entry for each rotor (rev/sec)

        # Goal: reach target position
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        # Reward inversely proportional to distance of target point
        reward = 1.-0.01*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        # Penalty for exiting boundary of space
        if self.sim.time < self.sim.runtime and self.sim.done == True:
            reward -= 10
    
        return reward

    # Accepts agent's choice of action (rotor_speeds), which is used to prepare the next state ot pass on to the agent
    # Reward is computed from get_reward()
    # Episode considered done if time limit is exceeded or quadcopter travelled outside bounds of simulation
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    # Agent should call this method every tim eepisode ends
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state