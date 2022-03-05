# Movie-review-classification-
a Write a code to extract TF-IDF features for each word from this dataset (remove
the labels which are the last entry of each line). Use the average the TF-IDF feature
as a document embedding vector (one feature per review) .
b Perform PCA on embedding vector to reduce to 10 dimensions.
c Train a two mixture diagonal covariance GMM on this data. Show the progress of
the EM algorithm by coloring each data point by assigning the data point to the
argmax of posterior probability of mixture component given the data point. Use the
first two PCA dimensions for this scatter plot.
d Check if the cluster identity of (max posterior probability of each review) correlates
with true label given for each review.
