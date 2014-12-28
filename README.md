2DToyWorld
==========

A simple tool for playing with 2D Classification, Regression, and Clustering problems. It is build ontop of JSAT and uses Java 8. 

This project is mostley meant as a simple means of playing with some algorithms on easy 2D data. I've been using this code for a while in testing / just playing around, and I've decided to open it up. However I wont be trying to keep the code as clean / maintained since its mostly just a toy for self learning and experimentation. 

When launched there is a simple dialog that gives lets you chose between the 3 types of problems. 

![Main Screen](https://github.com/EdwardRaff/2DToyWorld/raw/images/mainScreen.png )

Once you pick and option and a dataset is loaded, models and be selected from one of the menues. 

![Selecting Classifier](https://github.com/EdwardRaff/2DToyWorld/raw/images/selectingClassifierExample.png )

When you pick an option, if the model supports the "Paramaterized" interface of JSAT a dialog will show up with options to configure. 

![Selecting Classifier](https://github.com/EdwardRaff/2DToyWorld/raw/images/parameterExample.png )

Then you can play around and find a model that works well for your 2D toy problem! 

![Selecting Classifier](https://github.com/EdwardRaff/2DToyWorld/raw/images/svmRBFExample.png )

The Regression option dosn't support loading data - instead it has a few pre-made options to generate data from a 1D function. For Classification and Clustering you can load an ARFF file or a text file. With ARFF there must be exactly 2 numeric features in the file, and at least 1 nominal feature for Classification (it will be assuemd to be the target class). 

