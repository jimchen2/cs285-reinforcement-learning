1. Enforcing KL-Divergence Constraint with a Bonus

Problem:
Suppose we wish to learn $\pi$ under a KL-divergence constraint, i.e.,
$$ D(\pi, \pi_\beta ) = \mathbb{E}_{s \sim p_\pi} \left[ D_{KL}[\pi(a | s) \| \pi_\beta (a | s)] \right]. $$
How can we enforce this constraint by adding a bonus $b(s, a)$ to the reward $\bar{r}(s, a) = r(s, a) + \lambda b(s, a)$?

Solution:
$$ b(s, a) = \lambda \cdot D_{KL}[\pi(a|s) \| \pi_\beta(a|s)] $$


To enforce this constraint, we modify the reward function to include a bonus term that penalizes deviations from the baseline policy $$\pi_\beta$$. Specifically, the modified reward function $$\bar{r}(s, a)$$ is:

$$
\bar{r}(s, a) = r(s, a) + \lambda b(s, a),
$$

where $$\lambda$$ is a scaling factor that determines the strength of the penalty.

The bonus $$b(s, a)$$ is typically chosen to be proportional to the KL-divergence between $$\pi$$ and $$\pi_\beta$$ for the given state-action pair $$(s, a)$$. A common choice for the bonus is:

$$
b(s, a) = -D_{KL}[\pi(a | s) \| \pi_\beta (a | s)].
$$

Thus, the overall objective becomes:

$$
\bar{r}(s, a) = r(s, a) - \lambda D_{KL}[\pi(a | s) \| \pi_\beta (a | s)].
$$






2. Extending for f-Divergence

Problem:
The f-divergence is a generalization of the KL-divergence that can be defined for distributions P and Q by
$$ D_f [P \| Q] = \int Q(x)f\left(\frac{P(x)}{Q(x)}\right)dx $$
where $f$ is a convex function with zero at 1. We can state an f-divergence policy constraint as
$$ D(\pi, \pi_\beta ) = \mathbb{E}_{s \sim p_\pi} D_f [\pi(a | s) \| \pi_\beta (a | s)] = \mathbb{E}_{s \sim p_\pi} \mathbb{E}_{\pi_\beta (a|s)}\left[f\left(\frac{\pi(a | s)}{\pi_\beta (a | s)}\right)\right]. $$
How can you extend your answer from part (1) to account for an arbitrary f-divergence?

Solution:
To integrate an \( f \)-divergence constraint into a DQN framework, modify the reward function to include a bonus term reflecting the \( f \)-divergence. The general \( f \)-divergence between policies \( \pi \) and \( \pi_\beta \) is:

$$
D(\pi, \pi_\beta ) = \mathbb{E}_{s \sim p_\pi} \mathbb{E}_{\pi_\beta (a|s)}\left[f\left(\frac{\pi(a | s)}{\pi_\beta (a | s)}\right)\right].
$$

Adapt the reward function in DQN as follows:

$$
\bar{r}(s, a) = r(s, a) + \lambda b_f(s, a),
$$

where the bonus term \( b_f(s, a) \) is defined as:

$$
b_f(s, a) = -\mathbb{E}_{\pi_\beta (a|s)}\left[f\left(\frac{\pi(a | s)}{\pi_\beta (a | s)}\right)\right].
$$

In the DQN algorithm, this modification involves estimating the expectation under \( \pi_\beta \) and applying the function \( f \) to the ratio of action probabilities.


$$
\bar{r}(s, a) = r(s, a) + \lambda b_f(s, a),
$$

with \( b_f(s, a) \) as:

$$
b_f(s, a) = -\mathbb{E}_{\pi_\beta (a|s)}\left[f\left(\frac{\pi(a | s)}{\pi_\beta (a | s)}\right)\right],
$$



























3. Constraint for Finite Horizon and Trajectory Distributions

Problem:
Suppose $M$ has a finite horizon $H$ and we want to constrain divergence in the distribution of trajectories of states under $\pi$ and $\pi_\beta$. We can express the KL divergence between the (state) trajectory distributions for $\tau = (s_1, s_2, \ldots, s_H)$ as follows:
$$ D(\pi, \pi_\beta ) = D_{KL}[p_\pi (\tau) \| p_{\pi_\beta} (\tau)]. $$
What expression for $b(s, a)$ enforces this constraint? You may assume access to the dynamics $p(s'|s, a)$.

Solution:
$$ b(s, a) = \int p(s'|s, a) \cdot D_{KL}[p_\pi(\tau|s') \| p_{\pi_\beta} (\tau|s')] ds' $$

To enforce a constraint on the divergence between trajectory distributions under policies \( \pi \) and \( \pi_\beta \) in a finite horizon environment, the focus is on the KL divergence between these trajectory distributions. 


$$
D(\pi, \pi_\beta ) = D_{KL}[p_\pi (\tau) \| p_{\pi_\beta} (\tau)],
$$

the expression for \( b(s, a) \) is formulated as:

$$
b(s, a) = \int p(s'|s, a) \cdot D_{KL}[p_\pi(\tau|s') \| p_{\pi_\beta} (\tau|s')] ds'.
$$

 Calculating the KL divergence \( D_{KL}[p_\pi(\tau|s') \| p_{\pi_\beta} (\tau|s')] \) between the trajectory distributions under \( \pi \) and \( \pi_\beta \), conditioned on the next state \( s' \).

