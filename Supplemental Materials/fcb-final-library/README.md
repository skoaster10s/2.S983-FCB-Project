# FCB Python2 Model Library

This is the final Python2 library for the data processing and clustering analysis on La Liga 2016-17 event data.

## Getting Started

These instructions will give you the general flow of all the modules.

### Prerequisites

Other Python libraries needed to run this:

* matplotlib
* numpy
* pandas
* sklearn

### File Overviews

- **xml_to_csv.py**: script to convert the raw OPTA xml data into a workable csv form
- **csv_to_features.py**: script computing the feature vectors using the csv data in the data/ directory
- **best_k_analysis.py**: script to calculate and plot BIC scores and generate a t-SNE plot to determine best number of clusters to use
- **cluster_training.py**: script using csv_to_features functions to compute feature vectors for all the data and train a clustering model using the EM algorithm
- **cluster_results.py**: script using the trained model to compute results including team cluster distributions and most representative attacks from each cluster to analyze and interpret
- **filtering_results.py**: script to generate breakdowns for specific teams using the filtering functions

### Directory Overviews

- **data/**: contains the converted data being used from the original xml files
- **pickles/**: contains saved python objects used for the computations, including a saved clustering model, full dataset, and more

### Pickle File Overviews

- **attacks_full.pickle**: list of all attacks each represented by a dictionary of the necessary details
- **best_model_16.pickle**: trained GaussianMixture model from the sklearn library with k=16
- **team_attack_ranges.pickle**: dictionary mapping team ids to the index ranges containing that team's attacks in x_full.pickle and attacks_full.pickle
- **team_attacks.pickle**: dictionary mapping team ids to a list of all of that team's attacks represented as a dictionary of the necessary details
- **team_id_map.pickle**: dictionary mapping id number strings to team name strings
- **team_name_map.pickle**: dictionary mapping team name strings to id number strings
- **x_full.pickle**: numpy array of the full dataset, where each row consists of an attack's feature vector of length 21

-------------------
## How to Run

1. Run **xml_to_csv.py** on the OPTA xml dataset of choice to convert it to a compatible csv format. Be sure to specify location of data.
2. Run **cluster_training.py** on the converted csv data to create the necessary pickle files and train the initial model.
3. Run **best_k_analysis.py** to generate the plots for BIC scores for various values of k and t-SNE plot. Using these plots, determine the optimal number of clusters to be used. Currently, our model uses **K = 16**.
4. Rerun **cluster_training.py** to train a new model on a different value of K. (Can speed up the process by setting the boolean _data_from_pickle_ to True to load the data from pickles.)
5. Run **cluster_results.py** with the trained model to generate results on team breakdowns of attacks. Be sure to specify all parameters correctly.
6. Run **filtering_results.py** with the trained model and specified filter paramters to generate results on breakdowns in situations of interest.

