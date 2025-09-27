# POLICY ITERATION ALGORITHM

## AIM
The aim of this experiment is to implement the Policy Iteration Algorithm in Reinforcement Learning to determine the optimal policy and corresponding value function for a given environment. Policy Iteration combines iterative policy evaluation and policy improvement steps to achieve convergence towards an optimal policy.

## PROBLEM STATEMENT
In Reinforcement Learning, the agent interacts with an environment modeled as a Markov Decision Process (MDP).
The challenge is to find an optimal policy that maximizes the long-term cumulative reward.
Policy Iteration addresses this by:

Evaluating the value of a given policy (Policy Evaluation).
Improving the policy based on the evaluated value function (Policy Improvement).
Repeating these steps until the policy converges to the optimal policy.

## POLICY ITERATION ALGORITHM
# STEP 1:
Initialization

Initialize an arbitrary policy π and value function V(s).

# STEP 2:
Policy Evaluation

For the current policy π, compute the value function V(s) for all states until convergence.

# STEP 3:
Policy Improvement

Update the policy by choosing actions that maximize the expected return using the current value function.
# STEP 4:
Check for Convergence

If the policy does not change (π′ = π), then the policy is optimal and the algorithm terminates.
Otherwise, repeat steps 2 and 3.

## POLICY IMPROVEMENT FUNCTION
### Name: SASINTHARA S
### Register Number: 212223110045
```
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return new_pi

```
## POLICY ITERATION FUNCTION
### Name: POZHILAN V D
### Register Number: 212223240118
```
def policy_iteration(P,gamma=1.0,theta=1e-10):
  random_actions=np.random.choice(tuple(P[0].keys()),len(P))
  pi=lambda s: {s:a for s, a in enumerate(random_actions)}[s]
  while True:
    old_pi={s: pi(s) for s in range(len(P))}
    V=policy_evaluation(pi,P,gamma,theta)
    pi=policy_improvement(V,P,gamma)
    if old_pi=={s:pi(s) for s in range(len(P))}:
      break
  return V,pi

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy

<img width="516" height="165" alt="Screenshot 2025-09-27 102000" src="https://github.com/user-attachments/assets/a1153b1a-a9fd-4259-b68e-5c3cc1ef7596" />
<img width="543" height="177" alt="Screenshot 2025-09-27 102008" src="https://github.com/user-attachments/assets/a1722799-fdb8-4a1e-856b-87a0cbaeff08" />

### 2. Policy, Value function and success rate for the Improved Policy
<img width="553" height="173" alt="Screenshot 2025-09-27 101356" src="https://github.com/user-attachments/assets/28198f04-87e6-4ff8-b8e1-e67e86d20756" />
<img width="539" height="286" alt="Screenshot 2025-09-27 101408" src="https://github.com/user-attachments/assets/ef0fae95-ccf9-4ab7-9b39-c78bfa383c29" />


### 3. Policy, Value function and success rate after policy iteration

<img width="514" height="190" alt="Screenshot 2025-09-27 101433" src="https://github.com/user-attachments/assets/a2cd67e6-2ceb-4b07-bfca-3163da2fee85" />
<img width="539" height="166" alt="Screenshot 2025-09-27 101450" src="https://github.com/user-attachments/assets/f1164a54-51f0-4304-aa30-015d868f2cbd" />

## RESULT:
Therefore, policy iteration algorithm to find optimal policy by iteratively maximizing the value function is successfully implemented.
