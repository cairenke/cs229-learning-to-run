from osim.env.run import RunEnv

env = RunEnv(visualize=True, max_obstacles = 10)

observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    print('observation')
    print(observation)
    print('reward {0} done {1} info {2}'.format(reward, done, info))
    if done:
        env.reset()
#        break
    

