[<img src="https://img.shields.io/travis/playframework/play-java-starter-example.svg"/>](https://travis-ci.org/playframework/play-java-starter-example)

# SMS Classifier

This application performs sms classification using Weka and Play framework

## Dependencies

- [Weka](https://mvnrepository.com/artifact/nz.ac.waikato.cms.weka/weka-stable/3.8.0)

## Running

Run this using [sbt](http://www.scala-sbt.org/).

```
sbt run
```

And then go to http://localhost:9000 to see the running web application.

## Classifiers

- WekaClassifier.java

  Performs data loading,transformation, factorization, classifier building,classifier evaluation and prediction

## Controllers


- ClassificationController.java:

  Handles classification and Prediction using WekaClassifier.


## Components

- Module.java:

  Shows how to use Guice to bind all the components needed by your application.

## endpoints

- /train (GET)
    performs training on the training dataset

- /evaluate (GET)
    performs evaluation of the classifier using test data set

- /predict (GET) - query string message = <message to be classified>
    Performs classifcation of message

