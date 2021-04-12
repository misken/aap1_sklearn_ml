# Module 1 - Statistical/ML modeling
In the Intro to Data Science and/or Practical Computing for Data Analytics course, you learned how to create predictive models using techniques such as regression and tree-based methods. We will build on that knowledge and build classification models in Scikit-learn (sklearn) using a few different techniques, including logistic regression models with regularization - a technique for helping with variable selection and dimensionality reduction. learn about combining techniques into what are known as ensemble models. In addition, we’ll look at creating tree-based models within sklearn (both bagged and boosted). 

META: Picking up where we left off in pcda class. Leverages recently learned topics and much of this material is already developed and just needs cleaning and updating. For example, the regularization stuff is in R. We can weave the regularization in as part of the bundle of techniques used in the ensemble models. Let’s do classification. Gives gentle introduction to basic OO concepts but we aren’t really writing new OO code. Good place to introduce project structure, version control, and basic analysis pipelines. 

## Modeling/analysis objectives

* Review of scikit-learn library
* Advanced regression models (regularization)
* Bagged and boosted trees and general ensemble models

### Software engineering objectives
* Basics of conda virtual environments
* project structure (e.g. Cookiecutter data science)
* Version control with git and GitHub
* Good notebook practices
* Analysis pipelines 
* Deploy via web API or just hosted notebook




# Overall module flow

* Review of numpy, pandas and scikit-learn and intro to emsembles
  - the leaf classification problem, with modifications, is perfect for this
  - add a bit of numpy and pandas review (with pointers to the JVP stuff)
  - shows basic sklearn API (without pipelines) for quick modeling
  - meta bagging and voting classifiers
  - random forest as an ensemble model
      + **TODO** Screencast for the leaf notebook

* Introduce interesting classification problem - Pump It Up
* Structuring a data science project
  - folder structure
  - version control
  - good notebook practices
* Data prep - acquire, clean, transform, engineer features
  - 
* Model building 1 - individual techniques
  - review relevant techniques
  - build models
  - evaluate via cross-validation
  - predict on new data
  - evaluate for predictive accuracy using multiple metrics
* Building a modeling pipeline
* Deploying model as web service or in some other way (?) - binderizing might be good fit

## Introduce interesting classification problem

Let's use the [Pump it Up competition]() from [DrivenData.org](). I've already done a bunch of dataprep work and 

* it's a non-trivial and important problem.
* dominated by categorical data
* good opportunity for factor lumping and other feature engineering
* I did initial EDA in R, so chance to do in Python
* Doesn't require web scraping so can save that for another module
* Driven Data created cookiecutter-datascience template

Since this is an ongoing competition, we get training data that includes the target varaible values (i.e. we get $X$ and $y$) and then a set of test data with just the predictor values ($X$). For purposes of this analysis, we'll also resplit the training data to have our own test data for which we have the "answers". If we decide to actually make submissions in the competition, we will simply refit our models using all of the traning data.


## Project organization

Creating a custom [cookiecutter](https://cookiecutter.readthedocs.io/en/1.7.2/index.html) template for simple data science projects called `cookiecutter-datascience-simple`. It's available at https://github.com/misken/cookiecutter-datascience-simple. It's roughly based on the [cookiecutter-datascience template](https://drivendata.github.io/cookiecutter-data-science/) by [DrivenData.org](https://www.drivendata.org/) (and on various spinoffs of that template).

I created a new project called `pumpitupsk` by running:

    $ cookiecutter gh:misken/cookiecutter-datascience-simple/
    
which downloads the cookiecutter template and launches the quick start process. The resulting
project folder structure looks like:

    ── pumpitupsk
        ├── aap.yml
        ├── AUTHORS.rst
        ├── data
        │   └── raw
        ├── docs
        │   ├── conf.py
        │   ├── getting-started.rst
        │   ├── index.rst
        │   ├── make.bat
        │   ├── Makefile
        │   └── notes.md
        ├── LICENSE
        ├── notebooks
        │   └── template-nb.ipynb
        ├── output
        │   └── readme.md
        └── README.md
        
We'll do most of our work in the `notebooks/` directory.

**TODO** Create explanatory document of the cookiecutter template and add to the template itself and then reference here. Will force browsing of created project files and folders. Later in class we'll learn to
create a more complex template for projects in which we want to create a Python package.

### Resources

* https://pbpython.com/notebook-process.html

## Data acquisition

The data is available from the [Pump it Up project website](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/). To avoid everyone having to create an account, I've already downloaded the data and put it into `./data/raw`. There are three csv files:

* train_x.csv - predictor variables for the training data
* train_y.csv - target variables for the training data
* test_x.csv - predictor variables for the test data

Obviously, since this is an ongoing competition, the `test_y.csv` is not available. 

A description of the data is available at https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/.

## Data prep - cleaning, transformations, feature engineering

Let's create a new notebook for the data prep step. Opened the **template_nb.ipynb** notebook and saved as **data_prep.ipynb**.

## Model building

### Logistic regression with regularization

https://sklearn.org/modules/linear_model.html#logistic-regression


### Decision Trees

https://sklearn.org/modules/tree.html



### Ensemble methods

Random forests are a type of ensemble method -
https://sklearn.org/modules/ensemble.html#forests-of-randomized-trees

General bagging with an individual technique - 
https://sklearn.org/modules/ensemble.html#bagging-meta-estimator

Voting classifiers - 
https://sklearn.org/modules/ensemble.html#voting-classifier


```python

```
