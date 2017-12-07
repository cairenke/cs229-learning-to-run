from enrichedenv import EnrichedRunEnv

env = EnrichedRunEnv(visualize=False, max_obstacles = 3)

observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    # print('reward {0} done {1} observation {2}'.format(reward, done, observation))
    if done:
        env.reset()
        break
    

