# Module 4 Final Project


## Introduction

In this lesson, we'll review all of the guidelines and specifications for the final project for Module 4.

## Objectives

* Understand all required aspects of the Final Project for Module 4
* Understand all required deliverables
* Understand what constitutes a successful project

## Final Project Summary

Final module down -- you're absolutely crushing it! You've made it all the way through one of the toughest modules of this course. You must have an amazing brain in your head!

<img src='https://raw.githubusercontent.com/learn-co-students/dsc-mod-4-project-seattle-ds-102819/master/images/brain.gif'>

## The Datasets

For this module's final project, you have the choice of three different types of problems: 

- Time Series Modeling 
- Recommendation System 
- Image Classification with Deep Learning 
- Natural Language Processing


For each problem, we have provided a dataset.  You are highly encouraged to use the provided dataset, but you may use an alternative dataset of your own choosing, pending instructor approval.

Like Project \#3, the focus here is on *prediction*. It will be up to you to determine how best to evaluate your model, but for any of these projects you should imagine that your goal is to build something that **works**.  

When choosing a problem, consider:

1. **Portfolio Depth:** One option is to choose the same type of problem you plan to tackle in Module 5 (capstone).  This will allow you to practice the necessary skills in a group setting, before diving into your individual project.  You will likely produce a capstone project that is more polished and sophisticated, but your portfolio will demonstrate less breadth.
2. **Portfolio Breadth:** Another option is to choose a type of problem that interests you, but that you don't plan to use in your capstone project.  Each of your individual projects will end up less polished and sophisticated, but you will end up with a portfolio that demonstrates a wider range of skills.


### Problem 1: Time Series Modeling

If you choose the Time Series option, you will be forecasting real estate prices of various zip codes using data from [Zillow](https://www.zillow.com/research/data/). However, this won't be as straightforward as just running a time-series analysis -- you're going to have to make some data-driven decisions and think critically along the way!

For this project, you will be acting as a consultant for a fictional real-estate investment firm. The firm has asked you what seems like a simple question:

> What are the top 5 best zip codes for us to invest in?

This may seem like a simple question at first glance, but there's more than a little ambiguity here that you'll have to think through in order to provide a solid recommendation. Should your recommendation be focused on profit margins only? What about risk? What sort of time horizon are you predicting against?  Your recommendation will need to detail your rationale and answer any sort of lingering questions like these in order to demonstrate how you define "best".

As mentioned previously, the data you'll be working with comes from the [Zillow Research Page](https://www.zillow.com/research/data/). However, there are many options on that page, and making sure you have exactly what you need can be a bit confusing. For simplicity's sake, we have already provided the dataset for you in this repo -- you will find it in the file `time-series/zillow_data.csv`.

The goal of this project is to have you complete a very common real-world task in regard to time series modeling. However, real world problems often come with a significant degree of ambiguity, which requires you to use your knowledge of statistics and data science to think critically about and answer. While the main task in this project is time series modeling, that isn't the overall goal -- it is important to understand that time series modeling is a tool in your toolbox, and the forecasts it provides you are what you'll use to answer important questions.

In short, to pass this project, demonstrating the quality and thoughtfulness of your overall recommendation is at least as important as successfully building a time series model!

#### Starter Jupyter Notebook

For this project, you will be provided with a Jupyter notebook, `time-series/starter_notebook.ipynb`, containing some starter code. If you inspect the Zillow dataset file, you'll notice that the datetimes for each sale are the actual column names -- this is a format you probably haven't seen before. To ensure that you're not blocked by preprocessing, we've provided some helper functions to help simplify getting the data into the correct format. You're not required to use this notebook or keep it in its current format, but we strongly recommend you consider making use of the helper functions so you can spend your time working on the parts of the project that matter.

#### Evaluation

In addition to deciding which quantitative metric(s) you want to target (e.g. minimizing mean squared error), you need to start with a definition of "best investment".  Consider additional metrics like risk vs. profitability, or ROI yield.

### Problem 2: Recommendation System

If you choose the Recommendation System option, you will be making movie recommendations based on the [MovieLens](https://grouplens.org/datasets/movielens/latest/) dataset from the GroupLens research lab at the University of Minnesota.  Unless you are planning to run your analysis on a paid cloud platform, we recommend that you use the "small" dataset containing 100,000 user ratings (and potentially, only a particular subset of that dataset).

Your task is to:

> Build a model that provides top 5 movie recommendations to a user, based on their ratings of other movies. 

The MovieLens dataset is a "classic" recommendation system dataset, that is used in numerous academic papers and machine learning proofs-of-concept.  You will need to create the specific details about how the user will provide their ratings of other movies, in addition to formulating a more specific business problem within the general context of "recommending movies".

#### Collaborative Filtering

At minimum, your recommendation system must use collaborative filtering.  If you have time, consider implementing a hybrid approach, e.g. using collaborative filtering as the primary mechanism, but using content-based filtering to address the [cold start problem](https://en.wikipedia.org/wiki/Cold_start_(computing)).

#### Evaluation

The MovieLens dataset has explicit ratings, so achieving some sort of evaluation of your model is simple enough.  But you should give some thought to the question of metrics.  Since the rankings are ordinal, we know we can treat this like a regression problem.  But when it comes to regression metrics there are several choices: RMSE, MAE, etc.  [Here](http://fastml.com/evaluating-recommender-systems/) are some further ideas.

### Problem 3: Image Classification with Deep Learning

If you choose this option, you'll put everything you've learned together to build a deep neural network that trains on a large dataset for classification on a non-trivial task.  In this case, using x-ray images of pediatric patients to identify whether or not they have pneumonia.  The dataset comes from Kermany et al. on [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/3), although there is also a version on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) that may be easier to use.

Your task is to:

> Build a model that can classify whether a given patient has pneumonia, given a chest x-ray image. 

#### Aim for a Proof of Concept

With Deep Learning, data is king -- the more of it, the better. However, the goal of this project isn't to build the best model possible -- it's to demonstrate your understanding by building a model that works. You should try to avoid datasets and model architectures that won't run in reasonable time on your own machine. For many problems, this means downsampling your dataset and only training on a portion of it. Once you're absolutely sure that you've found the best possible architecture and other hyperparameters for your model, then consider training your model on your entire dataset overnight (or, as larger portion of the dataset that will still run in a feasible amount of time).

At the end of the day, we want to see your thought process as you iterate and improve on a model. A project that achieves a lower level of accuracy but has clearly iterated on the model and the problem until it found the best possible approach is more impressive than a model with high accuracy that did no iteration. We're not just interested in seeing you finish a model -- we want to see that you understand it, and can use this knowledge to try and make it even better!

#### Evaluation

Evaluation is fairly straightforward for this project.  But you'll still need to think about which metric to use and about how best to cross-validate your results.

### Problem 4: Natural Language Processing (NLP)

If you choose this option, you'll build an NLP model to analyze Twitter sentiment about Apple and Google products. The dataset comes from CrowdFlower via [data.world](https://data.world/crowdflower/brands-and-product-emotions). Human raters rated the sentiment in over 9,000 Tweets as positive, negative, or neither.

Your task is to:

> Build a model that can rate the sentiment of a Tweet based on its content.

#### Aim for a Proof of Concept

There are many approaches to NLP problems - start with something simple and iterate from there. For example, you could start by limiting your analysis to positive and negative Tweets only, allowing you to build a binary classifier. Then you could add in the neutral Tweets to build out a multiclass classifier. You may also consider using some of the more advanced NLP methods in the Mod 4 Appendix.

#### Evaluation

Evaluating multiclass classifiers can be trickier than binary classifiers because there are multiple ways to mis-classify an observation, and some errors are more problematic than others. Use the business problem that your NLP project sets out to solve to inform your choice of evaluation metrics.


## The Deliverables

Your completed project should contain the following four deliverables:

1. A well documented **_Executive Jupyter Notebook_** containing any code you've written for this project. This work will need to be pushed to a public GitHub repository dedicated for this project.

2. An organized **README.md** file in the GitHub repository that describes the contents of the repository. This file should be the source of information for navigating through the repository. 

4. An **_Keynote/PowerPoint/Google Slides Presentation_** that gives a brief overview of your problem/dataset, and each step of the CRISP-DM process.  The intended audience of this notebook is non-technical stakeholders.


### Excective Jupyter Notebook Must-Haves

For this project, your Jupyter Notebook should meet the following specifications:

#### Organization/Code Cleanliness

- The notebook is written for technical audiences with a way to both understand your approach and reproduce your results. The target audience for this deliverable is other data scientists looking to validate your findings.
- Well-organized, having distinct sections in logical sequence and with appropriate headings  
- Well-commented code, including doc-strings for functions
- Notebook can be run to generate same results. Includes setting random seeds & including referenced files in repository
- Text explains purpose of project and approach used and **decision making steps** throughout your project
- Text explains what code cells do
- Utilizes code for data cleaning through functions built elsewhere
- Text explains **reasoning** behind analytic decisions
- Text **interprets results** of analyses

#### Visualizations & EDA

* Your project contains at least 4 meaningful data visualizations, with corresponding interpretations. All visualizations are well labeled with axes labels, a title, and a legend (when appropriate)  
* Your notebook should contain 1 - 2 paragraphs briefly explaining your approach to this project.

#### Model Quality/Approach
- Pose at least one question that is relevant to the business problem and answer them through classification models
-  Runs 3 or more classification models, including justification for each model iteration
-  Models interations include hyperparameter tuning.
-  Models are validated
- Your notebook shows an iterative approach to modeling, and details the parameters and results of the model at each iteration.  
-  Whenever necessary, you briefly explain the changes made from one iteration to the next, and why you made these choices.  
-  You provide at least 1 paragraph explaining your final model.   

### Non-Technical Presentation Must-Haves

Another deliverable should be a Keynote, PowerPoint or Google Slides presentation delivered as a pdf file in your project repository with the file name of `presentation.pdf` detailing the results of your project.  Your target audience is non-technical people interested in using your findings to maximize their profit when selling their home.

Your presentation should:

* Contain between 5 - 10 professional-quality slides.  
    * **Level Up**: The slides should use visualizations whenever possible, and avoid walls of text.
* Take no more than 5 minutes to present.   
* Avoid technical jargon and explain the results in a clear, actionable way for non-technical audiences.  
* Clearly state the goals of the project
* Clearly describe the data used
* Relate the resulting analysis back to the organizational problem
* Include ideas to further develop the project 

**_Based on the results of your models, your presentation should discuss at least two concrete recommendations/solutions for the business problem._**


## Developing your Machine Learning Model

As you develop your machine learning model to address your selected problem mae sure to keep the following process, steps, and tools in mind.

![](Mod 4 Learning Goals.png)

## The Process

These steps are informed by the CRISP-DM process.

### 1. Getting Started

Please start by reviewing this document. If you have any questions, please ask them in Slack ASAP so (a) we can answer the questions and (b) so we can update this repository to make it clearer.

### 2. CRISP-DM Check-In pt.1 - Monday
You will meet with your instructors on Monday to “pitch” your project idea. We will be focusing on the steps of the CRISP-DM model.  Be prepared to answer the following questions. Be prepared to answer the following questions:


- Business Understanding
	- What is the business problem?
	- What questions will you be addressing?
   - Have you clearly defined your goal for your analysis?
   - Have your thought about who your audience is and how they would use this information?
   - How does this help the goals of the business/organization?
- Data Understanding
   - What data are you using?
   - How does your data help you answer the business question?
   - How many observations does your dataset contain?
   - What is the distribution of your data?
   - What data types do you have?   
- Data Preparation
   - Have you looked/dealt with missing values?
   - Have you done any data-type conversion?
      - ex: numerical data incorrectly ‘typed’ as strings.
   - Does your data contain any outliers or non-sensical values?
   -     
### 3. CRISP-DM Check-In pt.2 - Tuesday
- EDA/Visualization
   - What visualizations have you already made/planning to make?
   - What messages are these visualizations trying to convey?
   - What visualizations have you already made/planning to assess if your data meets the assumtions of linear regression?
   
- Modeling:
    - Is this a classification task? A regression task? Something else?
    - What models will we try?
    - How do we deal with overfitting?
    - Do we need to use regularization or not?
    - What sort of validation strategy will we be using to check that our model  - works well on unseen data?
    - What loss functions will we use?
    - What threshold of performance do we consider as successful?

- Evaluation:
    - How are you determining which variables are important.
    - What overall metric of success are you using.
    - What additional steps might you need to take to improve the model? (ex: transforming data, scaling data, getting more features) 
   
### 4. MVP Checkin - Wednesday
You will meet with an instructor who will check if the minimum requirements of the project are completed. AAt this point you should have a "best" model that "works" in answering the business question but may need some final adjustments.  Following this meeting you should be ready to polish your notebook and work on your slidedeck.


### 5. Practice Presentations- Thursday
You do a practice presentation in front of the class.  At this point the instructors and other students will provide verbal feedback which can be incorportated in your final presentation.

### 6. Project Presentations- Friday
You will present your project to the class using your slidedeck.  This presentation should not take more than 5 minutes and should be directed towards a non-technical audience.  Both partners should participate in the presentation of the project.

## Groups
1. James, Bobby, & Michael
2. Dan & Nick
3. Jim & Clair Marie
4. Jake & Alex
5. Raven & Sam

## Submitting your Project

You’re almost done! In order to submit your project for review, please complete the Mod 4 Project Submission form in Canvas.

## Grading Rubric

The grading rubric for the project can be found [here](https://docs.google.com/spreadsheets/d/1hbIZUQN2qipZZQsgMQRdBKvsTYRace4r09xkgKvmW_E/edit?usp=sharing).

## Summary

The end of module projects and project reviews are a critical part of the program. They give you a chance to both bring together all the skills you've learned into realistic projects and to practice key "business judgement" and communication skills that you otherwise might not get as much practice with.


