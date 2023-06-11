# Centro Universitário Senac
#  Bacharelado em Ciência da Computação
#  PI6 - Aplicação de Inteligência Artificial
#  Junho de 2023
#
# References and credits
# Gymnasium:
#   https://gymnasium.farama.org/content/basic_usage/

import gymnasium as gym

env = gym.make("ALE/Pitfall-v5", render_mode="human")
env.action_space = gym.spaces.Discrete(18)
observation, info = env.reset()

def main():
    # Configura a recompensa
    episode_reward = 0

    # Looping do jogo com duração estabelecida
    for _ in range(1000):
        action = env.action_space.sample()

        # Realiza uma ação no ambiente
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            print(f'Total reward: {episode_reward}')
            observation, info = env.reset()

if __name__ == '__main__':
    main()

env.close()
