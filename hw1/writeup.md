# Imitation Learning with Dagger

## Code Implementation Overview

<details>
<summary>Step the Env</summary>

```
ac = ptu.to_numpy(policy(ptu.from_numpy(ob.reshape(1, -1))).sample())
ac = ac[0]  # Retrieve the action value
next_ob, rew, done, _ = env.step(ac)  
```
</details>
<br/><br/>

<details>
<summary>Collecting Trajectories</summary>


```
paths, envsteps_this_batch = utils.sample_trajectories(env, actor, params['batch_size'], params['ep_len'])
```
</details>
<br/><br/>

<details>
<summary>Get Expert Policy</summary>


```
# Update actions in each path using expert policy
for path in paths:
    path["action"] = expert_policy.get_action(path["observation"])
```
</details>
<br/><br/>

<details>
<summary>Get Sampled Data</summary>


```
rand_indices = np.random.permutation(replay_buffer.obs.shape[0])[:params['batch_size']]
ob_batch, ac_batch = replay_buffer.obs[rand_indices], replay_buffer.acs[rand_indices]
```
</details>
<br/><br/>

<details>
<summary>Training the Actor</summary>


```
train_log = actor.update(ob_batch, ac_batch)
```
</details>
<br/><br/>

## Behavioral Cloning

### Ant
<details>

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1
```
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1 --batch_size 1000 --eval_batch_size 30000  --ep_len 1000 --n_layers 2 --size 32 --learning_rate 3e-2
```
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1 --batch_size 100 --eval_batch_size 30000  --ep_len 1000 --n_layers 2 --size 16 --learning_rate 4e-2
```
</details>
<br/><br/>

### HalfCheetah
<details>

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v4 --exp_name bc_halfcheetah --n_iter 1 --expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl --video_log_freq -1  
```

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v4 --exp_name bc_halfcheetah --n_iter 1 --expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl --video_log_freq -1 --batch_size 100 --eval_batch_size 30000  --ep_len 1000 --n_layers 3 --size 64 --learning_rate 2e-2
```
</details>
<br/><br/>


| Environment | Configuration | Mean | Standard Deviation |
|-------------|---------------|------|--------------------|
| Ant         | Naive Configuration | 3,401.8508 | 0 |
| Ant         | `--size 32 --learning_rate 3e-2` | 4,585.5176 | 108.1566 |
| Ant         | `--batch_size 100 --size 16 --learning_rate 4e-2` | 4,170.0972 | 829.4213 |
| HalfCheetah | Naive Configuration | 3,264.8667 | 0 |
| HalfCheetah | `--batch_size 100 --n_layers 3 --size 64 --learning_rate 2e-2` | 3,765.103 | 106.6637 |
## Dagger

Add`--do_dagger` to the end of each commands above and change iters
<!-- plotting the number of DAgger iterations vs. the policy’s
mean return,including behaviorr cloning performance and expert policy performance,show the standard deviation -->

### Ant

<details>
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1
```
</details>
<br/><br/>



<img src="https://github.com/jimchen2/nonimportant/assets/123833550/b152941e-b730-41cc-a1c9-898813d92388" alt="image" style="width: 100%;">

### HalfCheetah

<details>

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v4 --exp_name bc_halfcheetah --n_iter 10 --expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl --video_log_freq -1 --do_dagger
```
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v4 --exp_name bc_halfcheetah --n_iter 5 --expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl --video_log_freq -1 --batch_size 100 --eval_batch_size 10000  --ep_len 1000 --n_layers 3 --size 64 --learning_rate 1e-2 --do_dagger
```
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v4 --exp_name bc_halfcheetah --n_iter 10 --expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl --video_log_freq -1 --batch_size 10000 --eval_batch_size 10000  --ep_len 1000 --n_layers 2 --size 64 --learning_rate 4e-3 --do_dagger
```
</details>
<br/><br/>


<img src="https://github.com/jimchen2/nonimportant/assets/123833550/581de3c1-45da-47a9-9ca1-02027142b02a" alt="image" style="width: 100%;">
