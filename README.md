# Unsupervised Learning
by Vince Petaccio
### Exploring clustering and dimensionality reduction
Whereas supervised learning consists of mostly function approximation, unsupervised learning 
principally focuses on describing data. Here, two methods of unsupervised learning are explored: 
clustering and dimensionality reduction. 

Clustering is a useful unsupervised learning technique for machine learning operations including 
data mining and classification tasks. The input data is divided into a set of partitions, and 
each data point is assigned to its nearest partition, with the specific distance measurement 
representing an opportunity for the injection of domain knowledge.

Dimensionality reduction through feature selection or feature transformation is
useful in knowledge discovery and for addressing the “curse of dimensionality,”
which applies when a data set is highly dimensional but contains few examples
relative to this dimensionality, frustrating algorithms’ attempts to accurately explore
the feature and sample space.

See [the analysis](Clustering and Dimensionality Reduction Analysis.pdf) for a more in-depth review of the experimental setup and the results.

### Running the Code
This project has been implemented in Python. To run the code, navigate to the directory
containing it, and create a new Conda environment using the requirements.txt file:

conda create --name env_name --file requirements.txt

Next, activate the virtual environment:

source activate env_name
 
Finally, run each of the experiments described in the report by navigating to the directory
and running the experiment in unsupervised_learning.py using the run_experiments function:

run_epxeriments(experiment_number, data_set)

where experiment_number can be 0 to run all experiments, or an integer from 1 to 5 inclusive
to run the corresponding experiment. Note that runtime may be significant, up to 8 hours
per experiment, and that the experiments will save .png images to the disk. data_set should 
be set  to either 'forest_type' or 'cover_type' to choose the data set to run.

yummy.py is a custom plotting utility necessary to run plot_results.
