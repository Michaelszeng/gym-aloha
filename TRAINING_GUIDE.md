# PPO Training Guide for ALOHA Insertion Task

This guide explains how to train a PPO (Proximal Policy Optimization) policy for the peg insertion task.

## Quick Start

1. **Install training dependencies:**
```bash
pip install -r requirements-train.txt
```

2. **Start training with default settings:**
```bash
python train_ppo.py --total-timesteps 1000000 --log-dir logs/ppo_insertion
```

Or use the quickstart script:
```bash
bash train_ppo_quickstart.sh
```

3. **Monitor training progress:**
```bash
tensorboard --logdir logs/ppo_insertion/tensorboard
```
Then open http://localhost:6006 in your browser.

4. **Evaluate trained model:**
```bash
python evaluate_ppo.py --model-path logs/ppo_insertion/best_model/best_model.zip --n-episodes 10 --save-video
```

## Training Configuration

### Key Hyperparameters

The default configuration is tuned for vision-based manipulation:

- **Learning Rate**: `3e-4` - Standard for PPO
- **Number of Environments**: `4` - Parallel environments for faster data collection
- **Steps per Update**: `2048` - Collect 2048 steps per environment before updating
- **Batch Size**: `64` - Minibatch size for gradient updates
- **Epochs**: `10` - Number of optimization epochs per update
- **Entropy Coefficient**: `0.01` - Encourages exploration
- **Gamma**: `0.99` - Discount factor for future rewards

### Hardware Requirements

- **GPU**: Strongly recommended for both rendering (EGL) and training (PyTorch)
  - Training uses GPU for CNN policy
  - Rendering uses GPU via EGL for faster environment simulation
- **RAM**: ~16GB recommended for 4 parallel environments
- **Training Time**: 
  - 1M timesteps: ~6-12 hours on a modern GPU
  - 5M timesteps: ~1-2 days

### Adjusting Hyperparameters

For **faster training** (less sample efficient):
```bash
python train_ppo.py \
    --learning-rate 5e-4 \
    --n-envs 8 \
    --n-steps 1024 \
    --total-timesteps 2000000
```

For **more stable training** (more sample efficient):
```bash
python train_ppo.py \
    --learning-rate 1e-4 \
    --n-envs 2 \
    --n-steps 4096 \
    --batch-size 128 \
    --ent-coef 0.001 \
    --total-timesteps 5000000
```

### Reward Shaping

The environment provides sparse rewards (0-4):
- 1: Touching both peg and socket
- 2: Grasping both objects
- 3: Peg touching socket
- 4: Successful insertion

**Dense reward shaping is used during training** to make learning easier. The wrapper adds small bonuses for:
- Maintaining higher reward levels (+1% of current reward)
- Progressing to the next reward level (+0.1)

During evaluation, the original sparse rewards are used for fair benchmarking. This ensures the policy learns from dense feedback but is evaluated on the true task performance.

## Network Architecture

The policy uses a custom CNN feature extractor:

1. **Input**: 480x640 RGB images from top camera
2. **CNN Backbone**:
   - Conv2d(3 → 32, kernel=8, stride=4) + ReLU
   - Conv2d(32 → 64, kernel=4, stride=2) + ReLU
   - Conv2d(64 → 64, kernel=3, stride=1) + ReLU
   - Flatten
3. **Feature Layer**: Linear(n_flatten → 512) + ReLU
4. **Policy Head**: 2x256 hidden layers
5. **Value Head**: 2x256 hidden layers (separate from policy)

Total parameters: ~10-15M depending on input resolution

## Training Outputs

During training, the following are saved to `--log-dir`:

- `checkpoints/`: Model checkpoints every `--checkpoint-freq` steps
- `best_model/`: Best model based on evaluation performance
- `final_model/`: Final model after training completes
- `tensorboard/`: TensorBoard logs for visualization
- `eval/`: Evaluation results

## Monitoring Training

### TensorBoard Metrics

Key metrics to monitor:

- **rollout/ep_rew_mean**: Average episode reward (target: 4.0 for success)
- **rollout/ep_len_mean**: Average episode length
- **train/policy_loss**: Policy network loss
- **train/value_loss**: Value network loss
- **train/entropy_loss**: Entropy of action distribution
- **eval/mean_reward**: Evaluation performance on original rewards

### Expected Learning Curve

With the sparse rewards, expect:
- **0-100k steps**: Random exploration, low rewards (0-1)
- **100k-300k steps**: Learning to grasp objects (reward ~2)
- **300k-800k steps**: Learning to align and touch (reward ~3)
- **800k+ steps**: Learning successful insertion (reward ~4)

Success is not guaranteed with 1M steps due to sparse rewards. Consider:
- Running longer (5-10M steps)
- Using reward shaping
- Curriculum learning (start with easier tasks)

## Evaluation

Evaluate your trained model:

```bash
# Basic evaluation
python evaluate_ppo.py --model-path logs/ppo_insertion/best_model/best_model.zip --n-episodes 10

# With video recording
python evaluate_ppo.py --model-path logs/ppo_insertion/best_model/best_model.zip --n-episodes 5 --save-video
```

Evaluation metrics:
- **Success Rate**: Percentage of episodes reaching reward 4
- **Mean Reward**: Average episode reward
- **Episode Length**: Number of steps per episode

## Troubleshooting

### Training is slow
- Check GPU usage: `nvidia-smi` (should see Python processes)
- Reduce number of environments: `--n-envs 2`
- Reduce image resolution in environment creation
- Use CPU rendering: Change `MUJOCO_GL=osmesa` (slower but less GPU memory)

### Policy not learning
- Train longer (5-10M timesteps) - sparse rewards require patience
- Increase exploration: `--ent-coef 0.05`
- Reduce learning rate for more stable learning: `--learning-rate 1e-4`
- Use more parallel environments: `--n-envs 8`
- Check TensorBoard for NaN values or exploding gradients

### Out of memory errors
- Reduce number of environments: `--n-envs 2`
- Reduce batch size: `--batch-size 32`
- Reduce CNN features: Modify `features_dim` in code
- Use CPU device: `--device cpu` (much slower)

### GPU rendering not working
- Check EGL installation: `python -c "import mujoco; mujoco.MjModel.from_xml_string('<mujoco/>')"`
- Try CPU rendering: `os.environ["MUJOCO_GL"] = "osmesa"`
- Check NVIDIA drivers: `nvidia-smi`

## Advanced: Custom Observations

To include proprioceptive information (joint positions), modify the environment:

```python
env = gym.make(
    "gym_aloha/AlohaInsertion-v0",
    obs_type="pixels_agent_state",
    render_mode="rgb_array"
)
```

This will include joint positions in the observation, which may help learning.

## Advanced: Curriculum Learning

The training already uses dense reward shaping while evaluation uses original rewards. This is a form of curriculum learning where:

1. The policy learns with dense feedback during training
2. Performance is measured on the original sparse task during evaluation
3. The policy naturally learns to optimize for the true objective

If you want to fine-tune on pure sparse rewards, you can disable the wrapper by modifying `train_ppo.py` to not use `RewardShapingWrapper` for training environments.

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [ALOHA Project](https://tonyzhaozh.github.io/aloha/)

