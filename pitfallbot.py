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
#
#   Digitalocean:
#       https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym

import gymnasium as gym

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

def main():
    observation, info = env.reset(seed=10)
    # Configura a recompensa
    episode_reward = 0
    rewards = []

    # Looping do jogo com duração estabelecida
    for _ in range(1000):
        
        action = env.action_space.sample()
        # Realiza uma ação no ambiente
        observation, reward, terminated, truncated, info = env.step(action)

        # Imprime a recompensa de acordo com a ação atual
        reward_sign = "+"
        if reward < 0:
            reward_sign = "-"
        print(f'{reward_sign} Reward: {reward} -> current action: {action_list[action]}')

        episode_reward += reward
        if terminated or truncated:
            rewards.append(episode_reward)
            break
            # observation, info = env.reset()

    print("\nAvarage reward: %.2f\n" % (sum(rewards) / len(rewards)))


if __name__ == '__main__':
    main()

env.close()
