# cs348k-card-mechanics

# Draft:
Overview

This project studies how to design small sets of card-game mechanics that compose into interesting gameplay. Inspired by games like Balatro and Slay the Spire, I will build a lightweight simulation framework for a Balatro-style card game, define a constrained mechanic space, and evaluate candidate mechanic sets using automated bot play rather than relying only on subjective playtesting.

The main goal is not just to add content, but to build a system for reasoning about game design choices quantitatively. If feasible, I will also implement the resulting mechanics as a mod, but the simulator and evaluation framework are the primary deliverables.

Problem

Games with highly composable mechanics can create rich strategy spaces, but designing those mechanics is difficult because interactions are hard to reason about and playtesting is expensive. This project asks:

How can we define and evaluate a small space of card-game mechanics so that we can measure whether they produce interesting, balanced, and varied gameplay?

Project Idea

I will design a small family of new mechanics for a Balatro-style card game and evaluate them in a simplified simulator. The simulator will include:

a barebones game-state representation
a rules engine for card and mechanic interactions
bot agents with explicit choice heuristics
seeded simulation runs for reproducibility
metrics for comparing mechanic sets

If time permits, I will additionally implement the best mechanic subset as a mod in an existing game environment.

Planned System
1. Simplified game model

I will create a minimal but expressive state representation that captures the core parts of the game needed for strategic decision-making, such as hand state, score state, deck progression, and active mechanic effects.

2. Mechanic space

I will define a constrained set of new mechanics, likely centered on a few interaction themes rather than a large number of unrelated effects. The goal is to make the design space small enough to evaluate carefully while still allowing interesting combinations.

3. Bot harness

I will implement bots that choose actions according to defendable heuristics. This will allow repeated experiments over many random seeds and will act as a substitute for large-scale playtesting in the early evaluation phase.

4. Evaluation framework

I will compare mechanic sets using repeated simulations and quantitative metrics.

Evaluation

I plan to evaluate mechanic sets using some combination of the following:

win rate across seeds
score distribution and variance
run length / game length
build diversity, such as how many distinct strategies appear
stability / robustness across random seeds
sensitivity to bot policy choices
evidence of dominant or degenerate strategies

The goal is to argue not just that a mechanic is strong, but whether it creates a healthy and interesting interaction space.

Success Criteria

A successful project will produce:

A working simulator for a Balatro-style card game loop
A small, clearly defined mechanic design space
A reproducible bot-play evaluation harness
Quantitative comparisons between mechanic sets
A clear argument for which mechanics compose well and why

A stretch goal is to implement the best mechanic subset as a playable mod.

Why this fits CS348K

This project is about system design for interactive mechanics. The contribution is not only a game mod or a set of cards, but a framework for specifying constraints, simulating outcomes, and evaluating design choices quantitatively. It treats game mechanics as composable components and studies what makes that composition successful.

# Expected Deliverables
simulator code
mechanic definitions
bot policies
experiment scripts
plots/tables of evaluation results
short writeup of findings
optional mod implementation

# Repo Structure
.
├── README.md
├── src/
│   ├── game_state.py
│   ├── rules.py
│   ├── mechanics.py
│   ├── agents.py
│   └── simulate.py
├── experiments/
│   ├── configs/
│   ├── run_experiments.py
│   └── analysis.ipynb
├── results/
├── docs/
└── mod/   # optional stretch goal

# Milestones
Milestone 1

Define the simplified game state and implement the core simulation loop.

Milestone 2

Implement an initial mechanic family and baseline bot policies.

Milestone 3

Run seeded experiments and collect evaluation metrics.

Milestone 4

Refine mechanic designs and analyze tradeoffs.

Milestone 5

Optional: implement the strongest mechanic subset as a mod.

Current Status

Proposal stage. Initial design work is focused on:

defining the game state
identifying the mechanic space
deciding what metrics best capture "interesting" gameplay
defining bot choice heuristics and constraints
Notes

This repo may evolve in one of two directions:

a standalone simulator only
a simulator plus a mod implementation

The simulator and evaluation framework are the core project regardless of whether the mod is completed.
