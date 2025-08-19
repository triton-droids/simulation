# Background: Behavior Cloning (BC)

Behavior cloning is a **supervised learning approach** for imitation learning, where the goal is to train an agent to mimic an expert's behavior by learning a mapping from states to actions `s → a`. 

You have access to a dataset of expert demonstrations:

$D = \{(s_i, a_i)\}_{i=1}^N$

where $s_i$ is the observed state and $a_i$ is the expert action. The objective is to learn a **policy** $\pi_\theta(a \mid s)$ parameterized by $\theta$ such that, given a new state, the policy outputs an action similar to what the expert would choose.

---

## Mathematical Formulation

For a deterministic policy, we have:

$\pi_\theta(s) = \hat{a}$

and we can train it with **mean squared error (MSE)**:

$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^N \| \pi_\theta(s_i) - a_i \|^2$

**Intuition:** The network is encouraged to output actions that are **close to the expert’s actions** in an L2 sense.

---

## Challenges with Deterministic MSE
MSE penalizes the squared distance between the network's output and the observed action. Implicitly, this assumes there is one "correct" action per state. If the expert takes different valid actions for the same state across demonstrations, MSE tries to minimze the average squared error across all examples, potentially producing invalid actions. 



---

## Gaussian Actor: Probabilistic Behavior Cloning

Instead of predicting a single deterministic action, model the policy as a **probabilistic distribution**:

$\pi_\theta(a \mid s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s))$

Here, the network predicts both a **mean** $\mu_\theta(s)$ and a **variance** $\sigma_\theta^2(s)$ for each action dimension.  

The training loss is the **negative log-likelihood (NLL)** of the expert action:

$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^N \log \mathcal{N}(a_i \mid \mu_\theta(s_i), \sigma_\theta^2(s_i))$

Or explicitly for Gaussian outputs:

$\mathcal{L}(\theta) = \frac{1}{2} \sum_{i=1}^N \left[ \frac{\|a_i - \mu_\theta(s_i)\|^2}{\sigma_\theta^2(s_i)} + \log \sigma_\theta^2(s_i) \right] + \text{constant}$

---

## Negative Log-Likelihood (NLL) with a Gaussian Actor

A **Gaussian actor** learns to predict a **distribution over actions** for each state rather than a single deterministic action. This allows the policy to capture the variability in expert demonstrations instead of averaging multiple plausible actions, which can produce invalid results.  

Training uses the **negative log-likelihood (NLL)**:

- The network predicts a **mean** $\mu_\theta(s)$ and **variance** $\sigma_\theta^2(s)$ for the action.
- NLL minimizes the “cost” of the expert action under the predicted distribution.
- Intuitively, this makes the **most probable actions for a given state more likely**, while also accounting for variability and uncertainty in the demonstrations.


**Key idea:** NLL encourages the policy to assign high probability to expert actions while allowing flexibility to model multiple plausible actions in the same state.


---

## Intuition

1. **Capturing multimodality:** The policy can **spread probability mass over multiple plausible actions**, instead of averaging them.
2. **Modeling uncertainty:** The variance allows the network to **express confidence**. Low variance for consistent expert behavior; high variance where the expert is variable.

---

## Challenges in Robotics

1. **Covariate shift / compounding errors**  
   - BC assumes the model sees states similar to the training data.  
   - Small errors during deployment may push the robot into **states not seen in demonstrations**, causing errors to compound.

2. **Multi-modality**  
   - Some tasks have multiple valid actions (e.g., multiple ways to grasp an object).  
   - Deterministic BC may average actions, producing invalid or suboptimal behavior.  
   - Using **stochastic policies** with Gaussian outputs can model multiple possibilities.

3. **Long-horizon tasks**  
   - Errors early in the trajectory can affect future states significantly.  


---