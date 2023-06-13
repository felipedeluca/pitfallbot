# Centro Universitário Senac
#  Bacharelado em Ciência da Computação
#  PI6 - Aplicação de Inteligência Artificial
#  Junho de 2023
#  Guilherme Lucioni
#  Felipe de Luca
#
# References and credits
#   Gymnasium:
#       https://gymnasium.farama.org/content/basic_usage/
#       https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
#
#   Digitalocean:
#       https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym
#
#   Real Python:
#       https://realpython.com/python-bitwise-operators/

import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import sys

action_list = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9:  "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE"
}

env = gym.make("ALE/Pitfall-ram-v5", render_mode="human", obs_type="ram")
env.action_space = gym.spaces.Discrete(18)

def create_default_dic():
    return np.zeros(env.action_space.n, env.action_space.dtype)

# https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
class PitfallBot:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.5
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        # self.q_values = defaultdict(np.zeros, {'shape': self.env.action_space.n, 'dtype': env.action_space.dtype})
        self.q_values = defaultdict(create_default_dic)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_best_action(self, game_state: tuple[int, int, int]):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # Retorna uma ação aleatória de acordo com a probabilidade epsilon
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # Caso contrário, retorna o maior valor Q de acordo com o estado do jogo informado
        game_state_id = bit_convert(game_state)
        return int(np.argmax(self.q_values[game_state_id]))

    def update_qvalue(
        self,
        game_state: tuple[int, int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_game_state: tuple[int, int, int],
    ):
        """Updates the Q-value of an action."""
        # print("========>  self.q_values[next_game_state] : ", self.q_values[next_state_id])
        next_game_state_id = bit_convert(next_game_state)
        future_q_value = (not terminated) * np.max(self.q_values[next_game_state_id])
        game_state_id = bit_convert(game_state)
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[game_state_id][action]
        )

        self.q_values[game_state_id][action] = (
            self.q_values[game_state_id][action] + self.learning_rate * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def store_memory(self):
        print("\nSaving memory to disk...")
        with open('botmemory.pkl', 'wb') as f:
            global memory
            memory = self.q_values.copy()
            pickle.dump(memory, f)
            f.close()
            print("Memory stored on disk!\n")
    
    def load_memory(self):
        print("\nLoading memory from disk...")
        with open('botmemory.pkl', 'rb') as f:
            self.q_values = pickle.load(f)
            f.close()
            print("Memory loaded!\n")


def bit_convert(n):
    out = 0
    for bit in n:
        out = (out << 1) | bit
    return hash(out)

def train(bot: PitfallBot, num_episodes: int, num_actions: int):
    for episode in tqdm(range(num_episodes)):
        game_state, info = bot.env.reset()
        # Configura a recompensa
        episode_reward = 0
        rewards = []
        episode_done = False
        # hyperparameters
        # Looping do jogo com duração estabelecida
        # while not episode_done:
        for _ in range(num_actions):
            action = bot.get_best_action(game_state)
            # Realiza uma ação no ambiente
            next_game_state, reward, terminated, truncated, info = env.step(action)

            # Atualiza o bot
            bot.update_qvalue(game_state, action, reward, terminated, next_game_state)

            # game_state_id = bit_convert(game_state)
            # Imprime a recompensa de acordo com a ação atual
            reward_sign = "+"
            if reward < 0:
                reward_sign = "-"

            # print(f'{reward_sign} Reward: {reward:6} -> {action_list[action]:15} Game state: {game_state_id}')

            # Atualiza se o jogo atual terminou
            game_state = next_game_state
            done = terminated or truncated
            if done:
                break
        
        bot.decay_epsilon()


learning_rate = 0.01
num_episodes = 2
start_epsilon = 1.0
epsilon_decay = start_epsilon / (num_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
num_actions = 2000

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

bot = PitfallBot(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

bot.load_memory()

train(bot, num_episodes, num_actions)

bot.store_memory()

env.close()
