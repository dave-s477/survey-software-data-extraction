# Bi-LSTM-CRF for Software and Database recognition

In this repository we provide the code we used to train the bi-LSTM-CRF on the extraction of software and database names.

We based our model on the work of [Ma and Hovy 2016](https://arxiv.org/pdf/1603.01354.pdf) and an [implementation](https://github.com/guillaumegenthial/sequence_tagging) of their network in Tensorflow.
To keep it simple we did not perform exhaustive parameter search but used the hyper-parameters from the implementation or the article. 

Since this base model is based on word embeddings, we chose a word2vec model which was trained on scientific articles as described in [Pyssalo et al. 2013](http://bio.nlplab.org/pdf/pyysalo13literature.pdf).
We used the model trained on the largest corpus available from [here](http://bio.nlplab.org/) which was trained on PubMed abstracts, PMC full texts and a Wikipedia dump. 

## Data 
We used the largest corpus we found to be available by Duck et al. and downloaded it from its [Sourceforge repository](https://sourceforge.net/projects/bionerds/files/). 
Since the amount of data samples is quite small (85 articles, 2,573 software, 1,270 database) we do not follow the original data split used by Duck et al. but instead perform a 10-fold cross validation on the data. 

Since the applied method works on a sentence level rather than an article level we also transform the bag of articles into a bag of sentences for our evaluation.
We are aware that this can positively benefit the performance in the following case:
If a single articles mentions the same software multiple times the samples might be distributed to different folds in a bag of sentences which allows potentially more learning based solely on names. 
Multiple mentions of the same software in an article are generally not common but still present in the corpus by Duck et al. because they include articles from the domain of bioinformatics. 
Those articles sometimes introduce new software which is then mentioned multiple times in the corpus. 
For the artefact database this problem is less prominent.

Aside this distortion we believe that our evaluation will give a good indication of how well neural network based methods can perform on this extraction task, especially if they can be provided with a larger training corpus.  

## Training
For training the model we perform a downsampling of negative samples.
Because artefact mentions are very rare only a sparse amount of sentences contains actual mentions.
In order to construct batches that are likely to contain training data we randomly downsampled those "empty" samples but kept all containing mentions.