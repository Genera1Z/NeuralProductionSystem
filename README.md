# Neural Production System

Unofficial implementation of paper NeurIPS 2021 paper "[Neural Production Systems](https://proceedings.neurips.cc/paper/2021/hash/d785bf9067f8af9e078b93cf26de2b54-Abstract.html)" (NPS).

# Features

- supports three experiments provided by official source code:
  - ``coordinate_arithmetic``
  - ``sequential_arithmetic``
  - ``mnist_transformation``
- thoroughly refactored and re-implemented these models and datasets
  - remove more than 90% of code
  - refactored the building of models and datasets to the same style
  - boosted training and testing speed to some degree
- (hope) corrected almost all the gaps between the paper declaration and its official source code:
  - official q/k attention dose not use shared weights (partially corrected)
  - official "hidden states" dose not obtained by sequential model (todo)
  - parimay and contextual selections are not conducted as its algorithm describes (partially)

# References

- published paper and official code 1: [NeurIPS 2021 Supplemental](https://proceedings.neurips.cc/paper/2021/hash/d785bf9067f8af9e078b93cf26de2b54-Abstract.html)
- arXiv paper and official code 2: [anirudh9119 / neural_production_systems](https://github.com/anirudh9119/neural_production_systems)

# Comments

Do not hesitate to contact me if you want to discuss some ideas related to NPS with someone.
