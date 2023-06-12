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

env = gym.make("ALE/Pitfall-v5", render_mode="human", obs_type="ram")
env.action_space = gym.spaces.Discrete(18)

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


class PitfallAgent:
    pass


def bit_convert(n):
    out = 0
    for bit in n:
        out = (out << 1) | bit
    return out

def main():
    game_state, info = env.reset(seed=10)
    # Configura a recompensa
    episode_reward = 0
    rewards = []

    num_episodes = 1000
    # Looping do jogo com duração estabelecida
    for _ in range(num_episodes):
        action = env.action_space.sample()
        # Realiza uma ação no ambiente
        new_game_state, reward, terminated, truncated, info = env.step(action)

        game_state_id = bit_convert(game_state)
        game_state = new_game_state
        # Imprime a recompensa de acordo com a ação atual
        reward_sign = "+"
        if reward < 0:
            reward_sign = "-"
        
        print(f'{reward_sign} Reward: {reward:6} -> {action_list[action]:15} Game state: {game_state_id}')

        episode_reward += reward
        if terminated or truncated:
            rewards.append(episode_reward)
            break
            # observation, info = env.reset()

    print("\nAvarage reward: %.2f\n" % (sum(rewards) / len(rewards)))


if __name__ == '__main__':
    main()

env.close()
