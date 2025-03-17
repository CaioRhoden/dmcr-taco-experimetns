# Experiment: Context Relevance by Difficulty

## Goal

The objective of this experiment is to evaluate how different problems difficulties impact over the code generation performance

## Setup

The following specifications will be used in this experiment:
- The type of problem will be PROBABILITY
- It will be selected 5 different samples by difficulty on the test set
- For each sample will paired four other samples from the train set
- The matching algorith will be the cosine distance
- The model will be the Llama-3.2-Instruct-3B with the configurations:


## Metrics

- It will be used as measure the normalized absolute sum in the tests for each sample
- It will be also calculate the pass@1 metric

## How to run it

## Results