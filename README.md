# IDL_F24_Project_team_30
### Project Name
Robot Hand Manipulation Project

### Description
Implementation of PPO (Proximal Policy Optimization) and RSAC (Recurrent Soft Actor-Critic) algorithms for robotic hand manipulation tasks using the Gymnasium Robotics environment.

### Project Structure
    ├── RobotHand_Baseline.py   # PPO implementation

    ├── RobotHand_RSAC.ipynb       # RSAC implementation

    ├── Shadow_Hand_Example.ipynb   # Basic example using Shadow Hand

    ├── components/                 # Initial component implementations

            ├── GAE.py

            ├── model.py

            ├── ppo_agent.py

            └── PPO_Loss.py

### Requirements
gymnasium
gymnasium-robotics
mujoco
tensorboardX
torch
numpy
matplotlib
seaborn
pandas

### Features
- LSTM-based actor-critic architectures for temporal dependency modeling
- Experience replay buffer for sample efficiency  
- Comprehensive visualization and logging using TensorBoard
- Automated model checkpointing and early stopping
- Performance metrics tracking (rewards, success rate, consecutive successes)

### Training

#### PPO (RobotHand_Baseline.ipynb)
- Implements discretized action space
- Uses GAE (Generalized Advantage Estimation)
- Features clipped objective function and entropy regularization
- Includes gradient clipping for stability

#### RSAC (RobotHand_RSAC.ipynb)
- Continuous action space implementation
- Automatic entropy tuning
- Soft Q-learning with target networks  
- Recurrent state processing

### Usage

#### 1. Install dependencies:
```bash
pip install gymnasium gymnasium-robotics mujoco tensorboardX
```

#### 2. Run training:

For PPO: Execute RobotHand_Baseline.ipynb
For RSAC: Execute RobotHand_RSAC.ipynb

#### 3. Monitor training:

TensorBoard logs are saved in ppo_logs/ or rsac_baselogs/
Training curves and correlation matrices are auto-generated
Models are saved in models/ or models2/ directories

Model Outputs

Episode rewards
Average rewards
Success rates
Consecutive successes
Actor/Critic losses
Training metrics visualizations

Performance Evaluation

Mean reward
Success rate
Maximum consecutive successes
Episode lengths
Learning stability

Training automatically stops when achieving 50 consecutive successes or reaching the maximum episode limit.
