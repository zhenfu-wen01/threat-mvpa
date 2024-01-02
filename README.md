# threat-mvpa

This repository contains code and data for the paper: Distributed neural representations of conditioned threat in the human brain.

Install Python 3.8 and run 'pip -r install requirements.txt' to install the required toolboxes.

The main folder contains all the notebooks for the main analysis. The order to run the codes:
1. mvpa_step00_extract_features: extract features based on different masks;
2. mvpa_step01_cross_validation: conduct cross-validation on the discovery dataset;
3. mvpa_step02_external_validation: conduct external validation on the validation datasets;
4. mvpa_step03_predictive_patterns: estimate predictive patterns.


To replicate the figures in the paper:
1. mvpa_plot_decoding_accuracy: Fig. 1A-C and Fig. 2A-C
2. mvpa_plot_threat_circuit_patterns: Fig. 1D
3. mvpa_plot_whole_brain_patterns: Fig. 2D
4. mvpa_plot_generalization_accuracy: Fig. 3B-I

To apply the trained classifiers to external data:
mvpa_apply_classifier_to_your_own_data.ipynb

The masks folder contains all the masks used in the study.

The models folder contains all the trained models (classifiers).

The predictive_maps folder contains predictive patterns shown in Fig.1D and Fig. 2D.

The results folder contains data to produce the figures.

The sample_data contains sample data for running the analysis.

The sample_results contains temporary data generated by running the notebooks.
