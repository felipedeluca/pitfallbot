# pitfallbot_v2.py
#
# Centro Universitário Senac
#  Bacharelado em Ciência da Computação
#  PI6 - Aplicação de Inteligência Artificial
#  Junho de 2023
#  Guilherme Lucioni
#  Felipe de Luca
#
# References and credits for the source codes (partial or full):
#   Gymnasium:
#       https://gymnasium.farama.org/content/basic_usage/
#       https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
#
#   Digitalocean:
#       https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym
#
#   Real Python:
#       https://realpython.com/python-bitwise-operators/
#
#   Gelana Tostaeva:
#       https://medium.com/swlh/introduction-to-q-learning-with-openai-gym-2d794da10f3d

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import hashlib
from collections import defaultdict
from tqdm import tqdm


#Fixing seed for reproducibility
np.random.seed(0) 

action_list = {
    0:  ["NOOP", 0],
    1:  ["FIRE", 0],
    2:  ["UP", 0],
    3:  ["RIGHT", 10],
    4:  ["LEFT", -10],
    5:  ["DOWN", 0],
    6:  ["UPRIGHT", 0],
    7:  ["UPLEFT", 0],
    8:  ["DOWNRIGHT", 0],
    9:  ["DOWNLEFT", 0],
    10: ["UPFIRE", 0],
    11: ["RIGHTFIRE", 0],
    12: ["LEFTFIRE", 0],
    13: ["DOWNFIRE", 0],
    14: ["UPRIGHTFIRE", 0],
    15: ["UPLEFTFIRE", 0],
    16: ["DOWNRIGHTFIRE", 0],
    17: ["DOWNLEFTFIRE", 0]
}

# Initialize the Q table.
# Q_table = np.zeros((int, env.action_space.n), dtype=object)#defaultdict(create_default_dic)
Q_table = {"":[]}

# Filtro de memória, para remover da RAM do jogo dados como contador
# de tempo, entre outros, pois são considerados ruídos e
# dificultam a identificação do estágio e associação a melhor ação para
# determinado momento.
memory_filter = np.zeros(128, dtype=np.uint8)

def convert_to_id(data: dict):
    h = hashlib.md5(str(data).encode("utf-8"))
    id = str(h.digest())
    id = id.replace("\\", "")
    # print(id)
    return id


def get_additional_reward(action, state=[]):
    reward = 0
    # if len(state) > 0 :
    #     # Position 1 in the list means new scenario
    #     reward = state[1]
    reward = action_list[action][1]
    return reward


def add_new_entry(table: dict, key):
    if key not in table.keys():
        table[key] = np.zeros(18)


def policy_reward(action_sequence, current_reward) -> float:
    # Não interfere quando a recompensa pela ação é negativa. Isso evita
    # que o bot realize alguma ação que minimize ou neutralize a penalização.
    if current_reward < 0:
        return current_reward
    
    reward = current_reward
    action_count = len(action_sequence)
    if action_count < 2:
        return 0.0

    # Recompensa se a acao atual for diferente da última
    if (action_sequence[action_count-1] != action_sequence[action_count-2]):
        reward = 1.0
    
    # Recompensa se o personagem está avançando para a direita ou esquerda
    if (action_sequence[action_count-1] == 3 or action_sequence[action_count-1] == 4):
        reward += 1.0
    
    # Recompensa se subir a escada
    if action_sequence[action_count-1] == 2:
        reward += 1.0

    # Penaliza se descer a escada
    if action_sequence[action_count-1] == 5:
        reward = -2.0

    # Penaliza se ficar parado
    if action_sequence[action_count-1] == 0:
        reward = -1.0

    return reward


def calibrate_memory_filter():
    env = gym.make("ALE/Pitfall-ram-v5", render_mode=None, obs_type="ram", frameskip=1, repeat_action_probability=0.25)
    global memory_filter
    env.reset()[0]
    action = 0

    # Captura a tela inicial do jogo mantendo o personagem no lugar
    state, _, _, _, _ = env.step(action)

    filter_size = len(memory_filter)

    # Copia o conteúdo da tela inicial
    for i in range(filter_size):
        memory_filter[i] = state[i]
        # print(f'memory_filter[{i}] = {memory_filter[i]}')

    # Configura com valor zero todas as posições que tiverem valores diferentes
    for step in tqdm(range(1000), desc="Calibrating memory"):
        state, _, _, _, _ = env.step(action)
        for i in range(filter_size):
            if  memory_filter[i] != state[i]:
                memory_filter[i] = 1


def train(train_episodes: int, num_steps: int):
    env = gym.make("ALE/Pitfall-ram-v5", render_mode=None, obs_type="ram", frameskip=1, repeat_action_probability=0.25)
    env.action_space = gym.spaces.Discrete(18)

    # Check Gym environment state space
    print("\nGym Environment State Space")
    print("---------------------------")
    print(f'Action Space: {env.action_space}')
    print(f'State Space: {env.observation_space}\n')

    # Hyperparameters
    calibrate_memory_filter()
    
    learning_rate = 0.01
    discount_factor = 0.01
    exploration_rate = 1
    exploration_rate_max = 1
    exploration_rate_min = 0.01
    # Adapta a redução de aprendizado / exploração de acordo com o número de episódios
    exploration_decay_rate = exploration_rate_max / (train_episodes / 4)
    
    # Trainning the agent
    training_rewards = []
    exploration_rates = []    
    action = env.action_space.sample()

    recurring_states = 0
    states = 0

    global Q_table

    for episode in tqdm(range(train_episodes), desc="Episodes"):
        env.reset()
        action = env.action_space.sample()
        current_state, reward, done, truncated, info = env.step(action)
        total_training_rewards = 0
        lives = -1
        reward = 0.0
        # print(current_state)

        action_sequence = []

        for step in tqdm(range(num_steps), desc="Steps"):
            filter_memory_state(current_state)
            action_tradeoff = np.random.uniform(0, 1)
            current_state_id = convert_to_id(current_state)

            if current_state_id not in Q_table:
                add_new_entry(Q_table, current_state_id)
                action = 4
                Q_table[current_state_id][action] = reward

            # If tradeoff is larger than exploration_rate, then choose Exploitation
            if action_tradeoff > exploration_rate:
                action = int(np.argmax(Q_table[current_state_id][:]))
            # If tradeoff is smaller than exploration_rate, then choose Exploration
            else:
                action = int(env.action_space.sample())

            # Take the action and collect the reward and the outcome state
            new_state, reward, done, truncated, info = env.step(action)

            new_state_id = convert_to_id(new_state)

            # print(new_state)
            # Reward the character accordingly to the action 
            action_sequence.append(action)
            reward += policy_reward(action_sequence, reward)

            # print(new_state)

            if lives == -1:
                lives = info["lives"]
            else:
                if lives > info["lives"]:
                    reward -= 500
                    lives = info["lives"]

            # Update the equation using the Bellman equation
            terminated = truncated or done

            if new_state_id not in Q_table:
                add_new_entry(Q_table, new_state_id)
                Q_table[new_state_id][action] = reward

            total_training_rewards += reward
            future_q_value = (not terminated) * np.max(Q_table[new_state_id][:])
            temporal_difference = (reward + discount_factor * future_q_value - Q_table[current_state_id][action])
            Q_table[current_state_id][action] = Q_table[current_state_id][action] + learning_rate * temporal_difference
            current_state = new_state

            # print(f'REWARD = {reward}')

            # check if the episode is done
            if done or truncated:
                break

        # Lowering exploration by reducing the exploration_rate 
        exploration_rate = exploration_rate_min + (exploration_rate_max - exploration_rate_min) * np.exp(-exploration_decay_rate* episode)

        training_rewards.append(total_training_rewards)
        exploration_rates.append(exploration_rate)
    
    print ("\nTraining score over time: " + str(sum(training_rewards)/train_episodes))
    print (f'Total of recurring states: {recurring_states}\n')
    print (f'Total of new states: {states - recurring_states}\n')
    #Visualizing results and total reward over all episodes
   
    x = range(train_episodes)
    plt.plot(x, training_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Training total reward')
    plt.title('Total rewards over all episodes in training') 
    plt.show()

    #Visualizing the exploration_rates over all episodes
    plt.plot(exploration_rates)
    plt.xlabel('Episode')
    plt.ylabel('exploration_rate')
    plt.title("exploration_rate for episode")
    plt.show()


def store_memory(filename: str = 'botmemory'):
    print("\nSaving memory to disk...")
    fname = (f'{filename}.pkl')
    with open(fname, 'wb') as f:
        global Q_table
        pickle.dump(Q_table, f)
        f.close()
        print("Memory stored on disk!\n")


def load_memory(filename: str = 'botmemory'):
    print("\nLoading memory from disk...")
    fname = (f'{filename}.pkl')
    with open(fname, 'rb') as f:
        global Q_table
        Q_table = pickle.load(f)
        f.close()
        print("Memory loaded!\n")


def filter_memory_state(state):
    global memory_filter
    filter_size = len(memory_filter)

    for i in range(filter_size):
        if memory_filter[i] == 1:
            state[i] = 1


def play(num_steps: int):
    env = gym.make("ALE/Pitfall-ram-v5", render_mode="human", obs_type="ram", frameskip=1, repeat_action_probability=0.25)
    state = env.reset()[0]

    global Q_table

    for step in tqdm(range(num_steps)):
        filter_memory_state(state)

        state_id = convert_to_id(state)
        action = 0
        if state_id in Q_table.keys():
            action = np.argmax(Q_table[state_id][:])
            print(state_id)

        state, reward, done, truncated, info = env.step(action)

        if done or truncated:
            break


train_episodes = 1000
num_steps = 2000

train(train_episodes, num_steps)
store_memory()

load_memory()
play(num_steps)
