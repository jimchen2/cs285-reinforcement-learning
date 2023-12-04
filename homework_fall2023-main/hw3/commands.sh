# DQN
# Cartpole
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole.yaml 


# Lunar_Lander

python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 3
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_lr_0.05.yaml 


# double q
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 3




# Pacman
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman.yaml
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman_lr_3e-4.yaml
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman_lr_5e-4.yaml
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman_lr_5e-5.yaml








# SAC
# HalfCheetah
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reinforce1.yaml
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reinforce10.yaml
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reparametrize.yaml


# Hopper
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper.yaml
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper_clipq.yaml
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper_doubleq.yaml


# Humanoid
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/humanoid.yaml
