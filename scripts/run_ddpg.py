# Derived from keras-rl
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import argparse
from enrichedenv import EnrichedRunEnv
from extddpg import ExtDDPGAgent


def construct_model(env):
    nb_actions = env.action_space.shape[0]

    scalar = 1
    # Create networks for DDPG
    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(32 * scalar))
    actor.add(Activation('relu'))
    actor.add(Dense(32 * scalar))
    actor.add(Activation('relu'))
    actor.add(Dense(32 * scalar))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('sigmoid'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = Dense(64 * scalar)(x)
    x = Activation('relu')(x)
    x = Dense(64 * scalar)(x)
    x = Activation('relu')(x)
    x = Dense(64 * scalar)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    # Set up the agent for training
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)

    # set gamma to 0.995
    agent = ExtDDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.995, target_model_update=1e-3,
                      delta_clip=1.)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    return agent


def train(env, agent, num_timesteps, seed, model):
    agent.fit(env, nb_steps=num_timesteps, visualize=False, verbose=1, nb_max_episode_steps=env.timestep_limit,
              log_interval=10000)

    # After training is done, we save the final weights.
    agent.save_weights(model, overwrite=True)


def main():
    # Command line parameters
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
    parser.add_argument('--train', dest='train', action='store_true', default=True)
    parser.add_argument('--test', dest='train', action='store_false', default=True)
    parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
    parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
    parser.add_argument('--model', dest='model', action='store', default="example.h5f")
    parser.add_argument('--token', dest='token', action='store', required=False)
    parser.add_argument('--reward', dest='reward', action='store', default=0, type=int)
    args = parser.parse_args()

    # Load walking environment
    env = EnrichedRunEnv(args.visualize, 3, args.reward)

    # create model
    agent = construct_model(env)

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    if args.train:
        env.reset()  # difficulty = 2, seed = None)
        train(env, agent, args.steps, None, args.model)
    else:
        agent.load_weights(args.model)
        # Finally, evaluate our algorithm for 1 episode.
        env.reset(2, 1013)  # difficulty = 2, seed = None)
        agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=500)


if __name__ == '__main__':
    main()
