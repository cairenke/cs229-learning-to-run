from enrichedenv import EnrichedRunEnv

env = EnrichedRunEnv(visualize=True, max_obstacles = 3, reward_type=1)

level = 0
seed = 13
observation = env.reset(level, seed)
for i in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample())
    print('step {0} reward {1} distance {2}'.format(env.istep, reward, env.current_position))
    if done:
        env.reset(level, seed)
        break
    

