import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Define the environment
class GridEnvironment:
    def __init__(self, size=4, obstacles=None):
        self.size = size  # size x size grid
        self.start = 0
        self.goal = size * size - 1
        self.state = self.start
        # Define obstacle states
        if obstacles is None:
            self.obstacles = []
        else:
            self.obstacles = obstacles

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Actions:
        0: Up
        1: Down
        2: Left
        3: Right
        """
        row, col = divmod(self.state, self.size)
        original_state = self.state  # Keep track of original state

        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and row < self.size - 1:
            row += 1
        elif action == 2 and col > 0:
            col -= 1
        elif action == 3 and col < self.size - 1:
            col += 1

        next_state = row * self.size + col

        # Check if next_state is an obstacle
        if next_state in self.obstacles:
            next_state = original_state  # Stay in the same state
            reward = -1  # Negative reward for hitting an obstacle
            done = False
        elif next_state == self.goal:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.state = next_state
        return next_state, reward, done


# Q-Learning Agent
class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.Q = np.zeros((state_size, action_size))  # Initialize Q-table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit: choose the best action based on current Q-table
            return np.argmax(self.Q[state, :])

    def update_q(self, state, action, reward, next_state, done):
        if done:
            target = reward  # No next state
        else:
            target = reward + self.gamma * np.max(self.Q[next_state, :])
        # Q-learning update rule
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


# Visualization using matplotlib
class GridVisualizer:
    def __init__(self, env):
        self.env = env
        self.size = env.size
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_xticks(np.arange(0, self.size + 1, 1))
        self.ax.set_yticks(np.arange(0, self.size + 1, 1))
        self.ax.grid(True)
        self.ax.set_aspect("equal")

        # Draw goal
        goal_row, goal_col = divmod(env.goal, self.size)
        self.goal_patch = patches.Rectangle(
            (goal_col, self.size - goal_row - 1),
            1,
            1,
            linewidth=1,
            edgecolor="g",
            facecolor="green",
            alpha=0.5,
        )
        self.ax.add_patch(self.goal_patch)
        self.ax.text(
            goal_col + 0.5,
            self.size - goal_row - 0.5,
            "G",
            ha="center",
            va="center",
            fontsize=12,
            color="white",
        )

        # Draw obstacles
        self.obstacle_patches = []
        for obs in env.obstacles:
            obs_row, obs_col = divmod(obs, self.size)
            obstacle_patch = patches.Rectangle(
                (obs_col, self.size - obs_row - 1),
                1,
                1,
                linewidth=1,
                edgecolor="r",
                facecolor="red",
                alpha=0.5,
            )
            self.ax.add_patch(obstacle_patch)
            self.obstacle_patches.append(obstacle_patch)

        # Initialize agent's representation as a blue circle
        self.agent_circle = patches.Circle(
            (0.5, self.size - 0.5),
            0.3,
            linewidth=1,
            edgecolor="b",
            facecolor="blue",
            zorder=5,
        )
        self.ax.add_patch(self.agent_circle)

        plt.ion()
        plt.show()

    def update_agent_position(self, state):
        row, col = divmod(state, self.size)
        x = col + 0.5
        y = self.size - row - 0.5
        self.agent_circle.center = (x, y)
        plt.draw()
        plt.pause(0.05)  # Reduced pause time for faster animation

    def close(self):
        plt.ioff()
        plt.show()


# Training the agent with visualization
def train_agent(episodes=50):
    # Define obstacle states (for a 4x4 grid, states 5, 7, 10 are chosen as obstacles)
    obstacles = [5, 7, 10]
    env = GridEnvironment(size=4, obstacles=obstacles)
    agent = QLearningAgent(state_size=env.size * env.size, action_size=4)
    visualizer = GridVisualizer(env)

    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            # Update visualization
            visualizer.update_agent_position(state)

            if done:
                print(f"Episode {episode + 1}: reached goal in {steps} steps.")
                break

        agent.decay_epsilon()

    visualizer.close()
    return agent.Q


# Run the training
if __name__ == "__main__":
    final_q_table = train_agent(episodes=50)  # Adjust the number of episodes as needed
    print("\nFinal Q-Table:")
    np.set_printoptions(precision=2, suppress=True)
    print(final_q_table)
