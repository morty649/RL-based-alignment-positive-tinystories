# RL-Based Alignment on a Small GPT Model

Live at : 
  https://huggingface.co/spaces/morty649/positive-tinystories-gpt

  try it out for yourself !!

This repository implements a controlled reinforcement learning–based alignment experiment on a small character-level GPT model trained on TinyStories.

The goal is to study how policy gradient methods with KL regularization influence generative behavior under an automated reward signal.

This is not full RLHF.
No human preference data or reward model trained from comparisons is used.
Instead, alignment is performed using an automated sentiment-based reward.

# Summary

I trained a small GPT model (~1M parameters) using a three-stage pipeline:

Pretraining

Supervised Fine-Tuning (SFT)

Policy Gradient Reinforcement Learning

The RL phase optimizes a KL-constrained objective to increase output positivity while maintaining proximity to a pretrained reference model.

This repository is intended as a small-scale experimental environment for studying alignment dynamics.

Alignment Objective

During the RL phase, the following objective is optimized:
  
  L  =  −E[r⋅logπθ​]+βKL(πθ​∥πref​)

Where:
  r is an automated sentiment-based reward
  πθ​ is the current policy
  𝜋𝑟𝑒𝑓 is the frozen pretrained model
  β controls KL regularization strength

The reward term shifts model behavior toward higher positivity.
The KL term limits policy drift from the base distribution.

Model

Architecture: Character-level GPT

Parameters: ~1M

Context length: 64

Dataset: TinyStories

RL method: Vanilla policy gradient

Reward: Automated sentiment scoring

Motivation

Alignment techniques such as RLHF combine reward modeling with policy optimization.
This project isolates and studies the policy optimization component in a simplified setting.

Working at small scale enables:

Direct observation of reward shaping effects

Clear visualization of distributional drift

Controlled experimentation with KL strength

Rapid iteration without large compute requirements

# Results

Qualitatively, the RL-aligned model produces outputs with higher measured sentiment scores compared to the pretrained baseline.

Behavioral differences can be explored interactively:



# Installation
git clone https://github.com/morty649/RL-based-alignment-positive-tinystories.git
cd RL-based-alignment-positive-tinystories
pip install -r requirements.txt

# Train Pipeline 
 python3 tinystories_gpt_pg.py

# for viewing purposes run 

 python3 app.py
 open the localhost link in terminal and give me feedback 
