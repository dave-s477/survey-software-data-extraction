# Bi-LSTM-CRF for Software and Database Recognition

In this repository we provide the code we used to train the bi-LSTM-CRF on the extraction of software and database names.

We based our model on the work of [Ma and Hovy 2016](https://arxiv.org/pdf/1603.01354.pdf) and an [implementation](https://github.com/guillaumegenthial/sequence_tagging) of their network in Tensorflow.
To keep it simple we did not perform exhaustive parameter search but used the hyper-parameters from the implementation or the article. 

Since this base model is based on word embeddings, we chose a word2vec model which was trained on scientific articles as described in [Pyssalo et al. 2013](http://bio.nlplab.org/pdf/pyysalo13literature.pdf).
We used the model trained on the largest corpus available from [here](http://bio.nlplab.org/) which was trained on PubMed abstracts, PMC full texts and a Wikipedia dump. 

## Data 
We used the corpus by Duck et al. available from its [Sourceforge repository](https://sourceforge.net/projects/bionerds/files/). 
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

## To replicate our results in a 10-fold cross-validation: 
- Unpack the code 
- Install requirements: `pip install -r requirements.txt` (mostly [Tensorflow](https://www.tensorflow.org/), [sklearn](https://scikit-learn.org/stable/), [NLTK](https://www.nltk.org/), [pandas](https://pandas.pydata.org/) and the [tf_metrics](https://github.com/guillaumegenthial/tf_metrics))
- Run `python -m nltk.downloader 'punkt'` to download the NLTK data required for sentence tokenization.
- Download pre-trained word2vec word embedding `wikipedia-pubmed-and-PMC-w2v.bin` provided by Pyssalo ([http://bio.nlplab.org/](http://bio.nlplab.org/)) to folder `word_embeddings`. For example: `mkdir -p word_embeddings && cd word_embeddings && wget http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin`
- Download Duck et al. goldstandard from [https://sourceforge.net/projects/bionerds/files/goldstandard/goldstandard_all.zip/download](https://sourceforge.net/projects/bionerds/files/goldstandard/goldstandard_all.zip/download), create folder `goldstandard` with HTML files in base directory. For example: `mkdir -p goldstandard && cd goldstandard && wget https://sourceforge.net/projects/bionerds/files/goldstandard/eval.set2.25.human.html && wget https://sourceforge.net/projects/bionerds/files/goldstandard/eval.human.ascii.html && wget https://sourceforge.net/projects/bionerds/files/goldstandard/eval.next.5.human.ascii.html && wget https://sourceforge.net/projects/bionerds/files/goldstandard/gb.human.ascii.html && wget https://sourceforge.net/projects/bionerds/files/goldstandard/devel.human.ascii.html`
- run `python data_creation.py` (we assume python 3 and used version 3.7.3, no arguments required for our base setting). This step will transform the goldstandard into a suited input format for Tensorflow and place it in the folder `data`.
- run `python apply_model.py` (no arguments required for base settings). Exhaustive results are written in the folder which contains the base data, short summary is written to standard output (Tensorflow logging is outputted).
