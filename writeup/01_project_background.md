# 1. Project Background

### Abstract

This project explores the development and training of autonomous agents to play standard American Checkers using Reinforcement Learning (RL). The implementation evaluates two distinct RL paradigms across four major training configurations: (1) Proximal Policy Optimization (PPO) utilizing self-play, curriculum learning, and opponent pooling; (2) PPO integrated into a reward-diverse multi-agent league system (featuring tactical, terminal, and aggressive agents) to combat the passive co-evolution collapse observed during standard self-play; (3) an AlphaZero-style Monte Carlo Tree Search (MCTS) utilizing a standard scalar value head; and (4) an AlphaZero MCTS utilizing a Win/Draw/Loss (WDL) value head. Furthermore, to scale the training efficiently, I designed a centralized GPU inference server (utilizing my Nvidia RTX 5090) alongside parallel CPU workers, which significantly accelerated game simulation. This writeup documents the architecture, the infrastructure design, and the iterative debugging process I went through to build an agent skilled at playing checkers.

---

### Origin

The motivation for this project originated from observing the capabilities of Reinforcement Learning through creators like Code Bullet and the AlphaZero documentary on beating a Go grandmaster, as well as watching fascinating evolution and natural selection simulations by Primer. Seeing an agent start from zero knowledge and evolve complex, superhuman strategies through self-play and simulation demonstrated the power of these algorithms and inspired me to attempt building a similar system from the ground up, but much simpler. Hence, RL for checkers.

### First Attempt (Sophomore Year, Fall 2024)

I initially tackled this project during my sophomore year as the final project for my STAT 413 class (Statistical Machine Learning) at Rice University. The goal was to build a checkers environment and use it to train an agent using RL.

However, the implementation suffered from fundamental flaws due to my lack of RL knowledge. The agent's ability to learn was severely hampered by critical architectural errors. Key issues included the absence of proper loss penalties when losing, the use of ordinal, non-semantic action spaces (where identical network outputs mapped to entirely different board moves depending on the state), and silent bugs that flatlined the policy gradients (specifically, an action mask mismatch that collapsed the importance ratio, and a loop indentation error that threw away 99% of the collected training data before backpropagation). Compounding these issues was a lack of proper benchmarks and evaluation metrics, making it impossible for me at that time to diagnose these underlying bugs or even determine if the agent was learning. Ultimately, the agent failed to beat a random baseline. Acknowledging these limitations, I documented the findings for the final class submission and shelved the project.

### The Return (Junior Winter, 2026)

In the fall of my junior year, I took COMP 442 (Reinforcement Learning), which provided the theoretical foundation needed to understand my previous mistakes. Concepts such as Markov Decision Processes, policy gradients, and importance sampling clarified exactly why the previous architecture was mathematically incapable of learning.

Armed with this new knowledge, I restarted the project from scratch this winter while at my internship at RecruitU. This time, the objective was not just to train a capable agent at playing checkers, but to build robust, scalable training pipelines, and learn as much as possible. By developing custom parallelized infrastructure and methodically iterating through training configurations—starting from PPO with curriculum learning and opponent pooling, advancing to a reward-diverse PPO league system, and finally implementing and comparing AlphaZero MCTS with both scalar and WDL value heads—I am proud to say that I was able to achieve my goals. 

This writeup details the design principles, the engineering challenges, and the training insights from that effort.