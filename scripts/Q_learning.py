import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# Define the environment
class GridEnvironment:
    def __init__(self, size=4, obstacles=None):
        self.size = size  # size x size grid
        self.start = 0
        self.goal = size * size - 1
        self.state = self.start
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

        # Load agent PNG and wrap it for display
        self.agent_img = mpimg.imread("robi_mini.png")  # Ensure this file exists
        self.agent_imgbox = OffsetImage(self.agent_img, zoom=0.3)  # Scale image down
        # Initialize agent at start position
        start_row, start_col = divmod(env.start, self.size)
        start_x = start_col + 0.5
        start_y = self.size - start_row - 0.5
        self.agent_ab = AnnotationBbox(
            self.agent_imgbox,
            (start_x, start_y),
            frameon=False,
            zorder=5,
        )
        self.ax.add_artist(self.agent_ab)

        # Initialize Q-value text annotations for each state-action pair
        self.q_texts = {}
        for s in range(self.size * self.size):
            row, col = divmod(s, self.size)
            center_x = col + 0.5
            center_y = self.size - row - 0.5
            # Positions for each action relative to the center of the cell
            positions = {
                0: (center_x, center_y + 0.3),  # Up
                1: (center_x, center_y - 0.3),  # Down
                2: (center_x - 0.3, center_y),  # Left
                3: (center_x + 0.3, center_y),  # Right
            }
            for action, pos in positions.items():
                text_obj = self.ax.text(
                    pos[0],
                    pos[1],
                    f"{0.00}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    zorder=6,
                )
                self.q_texts[(s, action)] = text_obj

        plt.ion()
        plt.show()

    def update_agent_position(self, state):
        row, col = divmod(state, self.size)
        x = col + 0.5
        y = self.size - row - 0.5

        # Remove the old annotation
        self.agent_ab.remove()

        # Create a new annotation at the new (x, y)
        self.agent_ab = AnnotationBbox(
            self.agent_imgbox, (x, y), frameon=False, zorder=5
        )
        self.ax.add_artist(self.agent_ab)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.00001)

    def update_q_texts(self, Q):
        # Update the text annotations with new Q-values
        for (s, a), text_obj in self.q_texts.items():
            text_obj.set_text(f"{Q[s, a]:.2f}")
        self.fig.canvas.draw()

    def close(self):
        plt.ioff()
        plt.show()


# Training the agent with visualization
def train_agent(episodes=50):
    # Define obstacle states (for a 6x6 grid example)
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

            # Update visualization for agent movement
            visualizer.update_agent_position(state)

            if done:
                print(f"Episode {episode + 1}: reached goal in {steps} steps.")
                break

        # Decay exploration rate and update Q-values display at end of episode
        agent.decay_epsilon()
        visualizer.update_q_texts(agent.Q)

    visualizer.close()
    return agent.Q


# Run the training
if __name__ == "__main__":
    final_q_table = train_agent(episodes=50)  # Adjust the number of episodes as needed
    print("\nFinal Q-Table:")
    np.set_printoptions(precision=2, suppress=True)
    print(final_q_table)
