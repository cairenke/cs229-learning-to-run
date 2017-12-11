import argparse
import os

import tensorflow as tf
from baselines import logger
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple
from enrichedenv import EnrichedRunEnv
from mpi4py import MPI


def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=2)


def train(session, env, num_timesteps, model):
    # pposgd_simple.learn(env, policy_fn,
    #     max_timesteps=int(num_timesteps * 1.1),
    #     timesteps_per_actorbatch=256,
    #     clip_param=0.2, entcoeff=0.01,
    #     optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
    #     gamma=0.99, lam=0.95,
    #     schedule='linear'
    # )
    pposgd_simple.learn(env,
                        policy_fn,
                        max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=2048,
                        clip_param=0.2, entcoeff=0.01,
                        optim_epochs=10, optim_stepsize=1e-3, optim_batchsize=64,
                        gamma=0.99, lam=0.95,
                        adam_epsilon=1e-5,
                        schedule='linear')

    env.close()

    # After training is done, we save the final weights.
    if MPI.COMM_WORLD.Get_rank() == 0:
        saver = tf.train.Saver()
        saver.save(session, model)
        print('Saved model ' + model)


def main():
    session = U.single_threaded_session()
    session.__enter__()
    logger.session().__enter__()

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
    env.reset()  # difficulty = 2, seed = None)

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    if args.train:
        env.reset()  # difficulty = 2, seed = None)
        train(session, env, args.steps, args.model)
    else:
        observation = env.reset(0, 1013)
        pi = policy_fn('pi', env.observation_space, env.action_space)

        if os.path.exists(args.model + '.meta'):
            tf.train.Saver().restore(session, args.model)
            print('Loaded model %s' % args.model)
        else:
            print('Model %s not found' % args.model)
            exit(0)

        total = 0
        steps = 0
        done = False
        while not done:
            action = pi.act(True, observation)[0]
            observation, reward, done, info = env.step(action)
            total += reward
            env.render()
            steps += 1
        print('Total reward: %s' % total)
        print('Steps: %s' % steps)


if __name__ == '__main__':
    main()
