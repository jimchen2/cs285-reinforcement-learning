## Code Implementation (Compressed for Clarity)

### Sample Trajectory

<details>
Sample Trajectory

```
trajs, envsteps_this_batch = utils.sample_trajectories( env, agent.actor, args.batch_size, max_ep_len)
```

</details>
<br/>
<br/>

### Calculating q_values

<details>
Discounted Return

$$
\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_{it} | s_{it}) \sum_{t'=0}^{T-1} \gamma^{t'-1} r(s_{it'}, a_{it'})
$$

```
total_discounted_return = sum(self.gamma ** i * rewards[i] for i in range(len(rewards)))
discounted_rewards= [total_discounted_return] * n
```

Discounted Reward-to-go

$$
\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_{it} | s_{it}) \sum_{t'=t}^{T-1} \gamma^{t'-t} r(s_{it'}, a_{it'})
$$

```
running_add = 0
for t in reversed(range(len(rewards))):
    running_add = running_add * self.gamma + rewards[t]
    discounted_rewards[t] = running_add
```

`q_values = [self._discounted_return(r) for r in rewards]`

or

`q_values = [self._discounted_reward_to_go(r) for r in rewards]`

</details>
<br/>
<br/>

### Estimate Advantage

No baseline: `advantages = q_values`

Get critic values: `values = ptu.to_numpy(self.critic(obs).squeeze().detach())`

Baseline: `advantages = q_values - values`

Baseline with GAE:

$$
A^{\pi}_{\text{GAE}}(s_t, a_t) = \delta_t + \gamma \lambda A^{\pi}_{\text{GAE}}(s_{t+1}, a_{t+1})
$$

$$
A^{\pi}(s_t, a_t) \approx \delta_t = r(s_t, a_t) + \gamma V^{\pi}_{\phi}(s_{t+1}) - V^{\pi}_{\phi}(s_t)
$$

```
for i in reversed(range(batch_size)):
    delta = rewards[i] + self.gamma * values[i + 1] * (1 - terminals[i]) - values[i]
    advantages[i]=delta + self.gamma * self.gae_lambda * (1 - terminals[i])* advantages[i + 1]
```

### Normalizing and Updating

<details>
Normalizing:

```
if self.normalize_advantages:
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
```

Updating Actor

```
self.actor.update(obs=obs,actions=actions,advantages=advantages)
```

Updating Critic

```
for _ in range(self.baseline_gradient_steps):
    critic_info = self.critic.update(obs, q_values)
```

</details>
<br/>
<br/>

### Networks

<details>
<summary>Critic Network</summary>

Update Method

```
loss = F.mse_loss(self.forward(obs), q_values.view(-1, 1))
```

</details>
<br/>
<br/>

<details>
<summary>Policy Network</summary>

Forward (return distributions based on the observation)

```
if self.discrete:
    return distributions.Categorical(logits=self.logits_net(obs))
else:
    return distributions.Normal(self.mean_net(obs), torch.exp(self.logstd))
```

Update Method

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \pi_{\theta}(a_i | s_i) \cdot A(s_i, a_i) \right]
$$

```
distribution = self.forward(obs)
log_prob = distribution.log_prob(actions) if self.discrete else distribution.log_prob(actions).sum(axis=-1)
loss = -(log_prob * advantages).mean()
```

Then

$$
\nabla_{\theta} J(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ \nabla_{\theta} \log \pi_{\theta}(a_i | s_i) \cdot A(s_i, a_i) \right]
$$

```
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

</details>
<br/>
<br/>

## CartPole

<details>

```
# small batch
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na

#large batch
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na
```

</details>
<br/>
<br/>

<!-- ![Alt text](image-7.png) -->
<img src="https://github.com/jimchen2/nonimportant/assets/123833550/568d985d-3b9f-4bc9-9cc4-cf2a71ad6be3" /> 
<img src="https://github.com/jimchen2/nonimportant/assets/123833550/4a691e2a-ebe8-4328-935c-584481578462" /> 
<!-- ![Alt text](image-8.png) -->

- Which value estimator has better performance without advantage normalization?
- Without advantage normalization, rtg performs much better than default.
- Did advantage normalization help?
- Yes, advantage normalization helps reduce variance, leading to average returns being more stable
- Did the batch size make an impact?
- While small batch lead to fewer environment steps to converge, large batch has less variance after initially reaching 200.

## HalfCheetah

<details>

```
# No baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah
#add -na
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah_na -na
# Baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
# Baseline na
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline -na
# Customized
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 3 --exp_name cheetah_baseline_low_bgs
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.005 -bgs 5 --exp_name cheetah_baseline_low_blr
```

</details>
<br/>
<br/>

<img src="https://github.com/jimchen2/nonimportant/assets/123833550/85c01d4f-3397-4932-92c9-a274950e07ed" /> 
<img src="https://github.com/jimchen2/nonimportant/assets/123833550/2c94b1fa-407d-453d-a452-b8e98fd0f7d1" />

- Normalizing advantages makes learning faster.
- With a decreased number of bgs or blr, the half cheetah tends to be similar in performance.
- The default(without normalizing advantages or baseline) performs poor

## Inverted Pendulum

<details>
```
# finetuning 
    for seed in $(seq 1 5); do python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 200 --exp_name pendulum_default_s$seed --use_baseline -na -rtg --discount 0.95 --n_layers 2 --layer_size 16 --gae_lambda 0.98 --batch_size 1000 -lr 0.02 --seed $seed; done
```

</details>
<br/>
<br/>

<img src="https://github.com/jimchen2/nonimportant/assets/123833550/b4f0c5c3-6f41-4f6c-b3be-94aa9b9feef5" />

## LunarLander

<details>

```
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0    --exp_name lunar_lander_lambda0
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.95 --exp_name lunar_lander_lambda0.95
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.98 --exp_name lunar_lander_lambda0.98
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.99 --exp_name lunar_lander_lambda0.99
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 1.00 --exp_name lunar_lander_lambda1
```

</details>
<br/>
<br/>
<img src="https://github.com/jimchen2/nonimportant/assets/123833550/83e750c8-690c-4760-bfd5-838daf37a056" />

<!-- ![Alt text](image-4.png) -->

- λ = 0.95 (blue line) performs poorly, with low returns and high variance compared to other values.
- λ = 0.98 (orange line) and λ = 0.99 (green line) perform well, but λ = 0.99 experiences a significant drop towards the end.
- λ = 0 (red line) shows high variance and lower overall returns, suggesting that not accounting for future rewards is less effective. The advantage estimate becomes the same as the TD (Temporal Difference) error.
- λ = 1 (purple line) has an early rise in returns and exhibits high variance, indicating that overemphasizing long-term rewards can lead to unstable learning.

## Humanoid

<details>

```
python cs285/scripts/run_hw2.py --env_name Humanoid-v4 --ep_len 1000 --discount 0.99 -n 1000 -l 3 -s 256 -b 50000 -lr 0.001 --baseline_gradient_steps 50 -na --use_reward_to_go --use_baseline --gae_lambda 0.97 --exp_name humanoid --video_log_freq 5
```

</details>
<br/>
<br/>


<img src="https://github.com/jimchen2/nonimportant/assets/123833550/bd870e41-5e60-4c51-8094-446844d305ea" />

