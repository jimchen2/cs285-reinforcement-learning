## Training Loop

<details>
1. Compute Action \
`action = agent.get_action(observation, epsilon)`

- If `np.random.random()<epsilon`, Choose a random action \
`action = torch.tensor(random.randint(0, self.num_actions - 1))`
- Else choose from critic network \
`action = self.critic(observation).argmax(dim=1)`

2. Step environment 
3. Add data to replay buffer \
`replay_buffer.insert(...)`
4. Handle episode termination

5. Sample from replay buffer \
  `batch = replay_buffer.sample(config["batch_size"])`     
6. Train agent   \
 `agent.update(...)`
 - If `step % self.target_update_period == 0:`\
Update target network \
`self.target_critic.load_state_dict(self.critic.state_dict())
`
- Update critic network\
 `self.update_critic(obs, action, reward, next_obs, done)`
</details>
<br/><br/>

## Update Critic

<details>

1. Compute all options of q_values \
`next_qa_values = self.target_critic(next_obs)`
2. Choose q_values
- Double_q 
```
# Use critic network to update actions
next_actions = self.critic(next_obs).argmax(dim=1)
# Choose the Q values based on actions
next_q_values = next_qa_values.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
```
- Else choose the max Q values \
`next_q_values = next_qa_values.max(dim=1).values`
3. Compute target_values \
`target_values = reward+self.discount*next_q_values*(~done)`

4. Get q_values from critic network\
`q_values = self.critic(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)`

5. Compute loss function \
`loss = self.critic_loss(q_values, target_values)`
6. Update critic network 

  ```
  self.critic_optimizer.zero_grad()
  # Gradient clipping
  loss.backward()
  self.critic_optimizer.step()
  ```
</details>
<br/><br/>


## Experiments to Run 
<details>

```
# Cartpole
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole.yaml

# Lunar_Lander
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 3


# double q
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 3




# Pacman
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman.yaml
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman_lr_3e-4.yaml
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman_lr_5e-4.yaml
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman_lr_5e-5.yaml

```
</details>

<br/><br/>

## Results

<img src="https://github.com/jimchen2/nonimportant/assets/123833550/d3394894-2bd8-4b05-adad-696401bac6ac">
If learning rate of CartPole is too high then the predicted q values and critic error are both very high, leaing to overestimation.
<img src="https://github.com/jimchen2/nonimportant/assets/123833550/9f68d19f-20e5-4281-98b1-734c310dd7f7">
<img src="https://github.com/jimchen2/nonimportant/assets/123833550/3c251c9d-0633-4274-a90a-abe2f61a140b">
<img src="https://github.com/jimchen2/nonimportant/assets/123833550/8bb9ce61-0c73-4d99-b1ce-53bc4c621ebd">


