## Training Loop

<details> 
1. Compute Action

- Random sampling(at first) \
  `action = env.action_space.sample()`
- Then: `action = agent.get_action(observation)`

2.  Step environment \
    `...=env.step()`
3.  Add data to replay buffer \
    `replay_buffer.insert(...)`
4.  Handle episode termination
5.  Sample from replay buffer \
    `batch = replay_buffer.sample(config["batch_size"])`
6.  Train agent \
     `agent.update(...)`

    Update Critic \
    `self.update_critic(observations, actions, rewards, next_observations, dones)` 
    Update Actor \
    `self.update_actor(observations)` \
    Hard Update

    ```
    if step % self.target_update_period == 0:
        soft_update_target_critic(1.0)
    ```

    Soft Update
    `    self.soft_update_target_critic(self.soft_target_update_rate)`

    Update Function:
    `target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)`

</details>

<br/>
<br/>

Using a separate target network $Q_φ′$, perform soft updates
$$ φ′ ← φ′ + τ (φ − φ′) $$

## Entropy

Objective function for policy with entropy bonus.

$$ H(π(a|s)) = E_{a∼π} [− log π(a|s)] $$

In code: `-action_distribution.log_prob(action_distribution.rsample()).mean()`



## Update Critic

<details>

1. Get Actor Distribution `next_action_distribution = self.actor(next_obs)`
2. Sample from actor `next_action = next_action_distribution.sample()`
3. Get q_values `next_qs = self.q_backup_strategy(next_qs)`
</details>
<br/>
<br/>

- Double-Q
$$ y_A = r + γQ_{φ′_B} (s′, a′) $$
$$ y_B = r + γQ_{φ′_A} (s′, a′) $$
In code: `next_qs = torch.roll(next_qs, shifts=-1, dims=0)`
- Clipped double-Q:
$$ y_A = y_B = r + γ \min(Q_{φ′_A} (s′, a′), Q_{φ′_B} (s′, a′)) $$
In code: `next_qs = torch.min(next_qs, dim=0)[0]`

$$y ← r_t + γ(1 − d_t) [Q_φ(s_{t+1}, a_{t+1}) + βH(π(a_{t+1}|s_{t+1}))]$$

<details>


4. Compute Entropy(if used) \
               `next_action_entropy = self.entropy(next_action_distribution)`

    Then adding temperature \
    `next_qs -= (self.temperature * next_action_entropy)`


5. Compute the target Q-value \
            `target_values = reward + self.discount * (1- done) * next_qs`


6. Predict Q-values and Compute Loss

```
q_values = self.critic(obs, action)
loss = self.critic_loss(q_values, target_values)
```
</details>

<br/>
<br/>

## Update Actor

### REINFORCE

Actor with REINFORCE

$$ E_{s∼D,a∼π(a|s)} [∇_θ log(π_θ (a|s))Q_φ(s, a)] $$

<details>

1. Generate Action distribution \
   `action_distribution: torch.distributions.Distribution = self.actor(obs)`
2. Sample batch \
   `action = action_distribution.sample([self.num_actor_samples])`
3. Compute q_values
   `q_values = self.critic(obs, action)`

4. Compute loss  \
`loss = -torch.mean(q_values * action_distribution.log_prob(action))` 
5. Compute entropy \
`torch.mean(self.entropy(action_distribution))`
</details>
<br/>
<br/>

### REPARAMETRIZE

Actor with REPARAMETRIZE

$$ ∇_θ E_{s∼D, a∼π_θ(a|s)} [Q(s, a)] = ∇_θ E_{s∼D, ε∼N} [Q(s, μ_θ(s) + σ_θ(s)ε)] = E_{s∼D, ε∼N} [∇_θ Q(s, μ_θ(s) + σ_θ(s)ε)] $$

(Use rsample instead)

### Objective function for policy with entropy bonus.
$$ L_π = Q(s, μ_θ (s) + σ_θ (s)ε) + βH(π(a|s)) $$

In code: `loss -= self.temperature * entropy`


## Experiments to Run

<details>


```
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

```


</details>
<br/>
<br/>

## Results

![image](https://github.com/jimchen2/nonimportant/assets/123833550/d73b750e-0161-4b70-a9b3-bb5527510992)

![image](https://github.com/jimchen2/nonimportant/assets/123833550/4d080b90-7aba-4016-ba33-961c1a7a8c5a)
![image](https://github.com/jimchen2/nonimportant/assets/123833550/1207fdce-f26f-4a38-a710-5190667efd07)




The Q_values tend to be more stable with clipq. Singleq overestimates Q_values. Thus singleq tend todrop in performances.


![image](https://github.com/jimchen2/nonimportant/assets/123833550/ca47de8e-9173-4fef-980a-aba928d42861)

