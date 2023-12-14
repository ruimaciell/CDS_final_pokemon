# Final project of Computing for Data Science

<p align="center">
  <a href="https://pokemondb.net/pokedex/all/">
    <img src="https://github.com/ruimaciell/CDS_final_pokemon/blob/main/images/pokemon.png" alt="Header">
  </a>
</p>

<p align="center">
  <img src="https://github.com/ruimaciell/CDS_final_pokemon/blob/main/images/charmander.gif" alt="charmander">
</p>  
  
---
## Table of Contents

1. **Introduction**
    - [Overview of the Project](#overview-of-the-project)
    - [Installation and Onboarding](#installation-and-onboarding)
    - [Team Overview](#team-overview)

2. **Library Architecture**

3. **Pipeline Review**
    - [DataLoader (Presented by Mikel)](#dataloader-presented-by-mikel)
    - [*Bonus* Exploratory Data Analysis (Presented by Rui)](#bonus-exploratory-data-analysis-presented-by-rui)
    - [Preprocess and Feature Engineering (Presented by Rui)](#preprocess-and-feature-engineering-presented-by-rui)
    - [Model (Presented by Luis)](#model-presented-by-luis)

4. **Tests**

5. **Observations and Recommendations**
---
# Project Overview

Welcome to our Pokemon-inspired project! We've delved into the world of Pokemon, celebrated as one of the most cherished TV shows of all time.

## Objective

Our goal is to build a predictive model estimating a Pokemon's probability of winning in combat. Using a dataset featuring historical battle outcomes, attributes, and skills, we're decoding the intricacies of Pokemon battles.

## Dataset

We've provided four CSV files, with a focus on `combat.csv` and `pokemon.csv`. These files drive our presentation. While we envisioned future enhancements using `team_combats` and `team_pokemon_id`, time constraints during exams prevented us from realizing this part. Apologies for any inconvenience!

Embark on this Pokemon journey with us as we unravel the magic of battles and predictive modeling!





## Library Arquitecture  
<pre>
├── library_final
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-312.pyc
│   └── pokedex
│       ├── TBDload_class.py
│       ├── TBDmodel_class.py
│       ├── TBDprepro_class.py
│       ├── feature_class.py
│       ├── model_implementation.py
│       └── __init__.py
|
├── Notebooks
│   └── Main.ipynb
├── Test
├── images
│   ├── charmander.gif
│   └── pokemon.png   
├── raw_data
│   ├── ProcessedData.csv
│   ├── combats.csv
│   ├── pokemon.csv
│   ├── pokemon_id_each_team.csv
│   ├── team_combat.csv
│   └── your_dataset.csv
|
├── requirements.txt
├── README.md
├── setup.cfg
├── setup.py
├── testing.py
└── tree.txt 
  
6 directories, 23 files
</pre>