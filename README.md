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
    - [Project Overview](#roject-overview)
    - [Installation & Onboarding](#nstallation-&-onboarding)

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

---
# Installation and Onboarding

The installation of the library provides all the tools necessary for this model to run properly. Different functions, we decided to use classes for more clarity and interpretability, come into play in the model process at different stages. 
- First step is to Install the Library in your Local machine using `pip install -e .`
- Automate the loading of your dataset using the DataLoader module, which offers different functions that adapts to different working needs.
- Run your first exploratory data analysis using the EDA.py module
- A brief exploratory data analysis to visualize the information
- Creation of new features, special treatment of others and removal of unuseful
- Train and test split for training and prediction of the models.
- Machine Learning models applied to the data.



### Scalability and Robustness

Our approach to designing the process prioritizes generalization and readability. The avoidance of hardcoding and the maintenance of a clear structure play pivotal roles in ensuring scalability. Leveraging parent and child classes serves a dual purpose: it minimizes code repetition and facilitates the integration of processes, making it more adaptable to future modifications. A notable instance of this design philosophy is evident in the implementation of machine learning models.

In our system, a parent class acts as the overarching container for all models, efficiently calling methods specific to each model. Common processes such as fitting, training, predicting, and evaluating performance are shared across all models, contributing to a unified and streamlined structure. This design not only enhances code readability but also simplifies the incorporation of new machine learning models.

To introduce a new model, one only needs to create a new child class. This new class generates an instance of the model, handles the fitting process, and predicts the target variable. By referring to the methods established in the parent class, this approach eliminates the need to build an entirely new model from scratch, ensuring a scalable and robust foundation for future model expansions.

<img src="https://github.com/ruimaciell/CDS_final_pokemon/blob/main/images/image22.jpeg∫" alt="Header">

## Library Arquitecture  
<pre>
├── Notebooks
│   ├── Main_pokemon.ipynb
│   ├── Rui_presentation.ipynb
│   ├── query_credentials.json
│   └── raw_data
│       ├── ProcessedData.csv
│       ├── combats.csv
│       ├── pokemon.csv
│       ├── pokemon_id_each_team.csv
│       └── team_combat.csv
|
├── README.md
├── Test
│   │    
│   ├── test_DataPreprocessor.py
│   └── test_Feature.py
|
├── images
│   ├── charmander.gif
│   └── pokemon.png
|
├── library_final
│   ├── __init__.py
│   └── pokedex
│       ├── DataLoader.py
│       ├── DataPreprocessor.py
│       ├── EDA.py
│       ├── FeatureEng.py
│       ├── ModelSelector.py
│       ├── __init__.py
|
├── requirements.txt
├── setup.cfg
├── setup.py
└── tree.txt

12 directories, 47 files
</pre>


