# FDA
Functionnal Data Analysis
For deeper comprehension of the scikit-fda package, here is the tutorial developed by C RAmos-Carreño.
I do not own the skfda_tutorial repo on my github page and it's presence here is only as a tutorial.
Please for for a deeper dive, consult the skfda documentation website: https://fda.readthedocs.io/en/latest/index.html 

A simple implementation of a new convolutional architecture for FDA Neural networks.
TSC (the model we propose).
This model generalizes 1D conv for functional data and the limit model is analytic continuous convolution with the subdivision getting narrower.
Better than MLP to classify functional data.
Quicker in epoch and better for irregularly spaced functional data as it is based on B-spline basis expansion and modeling.
The functional .py file contains every utils function you need to automatically compare a wide variety of models for functional classification and regression. 
The class Hyperparameters is uselefull to set up the hyper parameters as you wish and its universal nature make it usable on different model classes.
Implemented for Gru lstm mlp 

