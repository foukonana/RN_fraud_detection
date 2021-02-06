## Anomaly Detection in Review Graphs ##

Solutions provided for anomaly detection with graphs are domain dependent. This is of course a corollary of the nature of outliers that are different and showcases different patterns in each domain.

In this project we are detecting anomalous users/reviews in review networks, without making use of review data in text format as there are many cases where reviews consist only of ratings and not text (content based approaches). We are interested mostly in the behavioral modeling that uses graph structure, as graph-based methods are able to leverage the relational ties between users, reviews, and products with minimal to no external information. 

We will be conducting a case study using YELP restaurant review network, using [BIRDNEST](http://www.alexbeutel.com/papers/birdnest_sdm16.pdf) and [SPEAGLE](http://shebuti.com/wp-content/uploads/2016/06/15-kdd-collectiveopinionspam.pdf) as they are currently the SOTA modelling for fraudulent user detection in review graphs.

Furthermore in depth graph analysis of YELP restaurant network is be provided. 

To get the datasets with ground truth please email miss Shebuti Rayana at: srayana@cs.stonybrook.edu

### Requirements ###
In order to reproduce this project you will need to create a python virtual environment. Because the original [BIRDNEST](http://www.alexbeutel.com/papers/birdnest_sdm16.pdf) code (can be found [here](https://bhooi.github.io/ratings.tar)) was written by the author in Python 2 we reproduced in the same version as transition to Python 3 wasn't straight forward. We have instructions for both python versions as there are pieces of code in this project written python 3.

* For Python 2
Create virtual env and activate it:
```sh
virtualenv fraud_detection_env2
source fraud_detection_env/bin/acticate
pip install -r requirements2.txt
```

* For Python 3
Create virtual env and activate it:
```sh
python3 -m venv fraud_detection_env3
source fraud_detection_env/bin/acticate
pip install -r requirements3.txt
```
*All scripts of the repo must be executed while being in the main directory
