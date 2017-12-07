from enrichedenv import EnrichedRunEnv

env = EnrichedRunEnv(visualize=False, max_obstacles = 3)

observation = env.reset()
for i in range(20):
    observation, reward, done, info = env.step(env.action_space.sample())
    print('reward {0} done {1} info {2}'.format(reward, done, info))
    if done:
        env.reset()
        break
    

