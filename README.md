# cs229-osim-rl
The Art of Human Movement

Getting started

Anaconda is required to run our simulations. Anaconda will create a virtual environment with all the necessary libraries, to avoid conflicts with libraries in your operating system. You can get anaconda from here https://www.continuum.io/downloads. In the following instructions we assume that Anaconda is successfully installed.

On Linux/OSX, run:

    conda create -n opensim-rl -c kidzik opensim git python=2.7
    source activate opensim-rl

These commands will create a virtual environment on your computer with the necessary simulation libraries installed. Next, you need to install our python reinforcement learning environment. Type (on all platforms):

    conda install -c conda-forge lapack git
    pip install git+https://github.com/stanfordnmbl/osim-rl.git
    conda install keras -c conda-forge
    pip install git+https://github.com/matthiasplappert/keras-rl.git
    cd [the directory where you have checked out the source code for this project]
    git install -e ./openai

If the command python -c "import opensim" runs smoothly, you are done! Otherwise, please refer to our FAQ section.

Note that source activate opensim-rl activates the anaconda virtual environment. You need to type it every time you open a new terminal.


Training your first model
Go to the scripts subdirectory for this project

    cd scripts

There are two scripts:

    run_ddpg.py for training (and testing) an agent using the DDPG algorithm.
    run_ppo.py for training (and testing) an agent using the PPO algorithm
    
Training

    python run_ddpg.py --visualize --train --model sample
    
    or
    
    mpirun -np 4 python run_ppo.py --train --model ppo1 --steps 10000

Test

    python run_ddpg.py --visualize --test --model sample
    
    or
    
    python run_ppo.py --visualize --test --model ./ppo1
