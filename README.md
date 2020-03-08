# DecisionTree-and-LogisticRegression
---
title: "DecisionTree and LogisticRegression"
author: "Doris Ying-Yu Kuo"
date: "12/7/2019"
output:
  html_document:
    toc: true
    toc_float: true
    theme: darkly
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# ● PART I: Collecting the Data

```{r}
mushroom = read.csv('https://s3.amazonaws.com/notredame.analytics.data/mushrooms.csv')
head(mushroom)
```
```{r, message=FALSE}
library(tidyverse)
```


```{r}
mushroom<-mushroom%>%
  mutate(edible=ifelse(type=='edible','Yes','No'))%>%
  mutate(edible=as.factor(edible))

#check
mushroom%>%
  select(edible,type)%>%
  head()

#delete type
mushroom<-mushroom%>%
  select(-type)
```

# ● PARTII: Explore and Prepare the Data 

## 1. Explore Data 

### Graphical Method 

```{r,fig.width=10, fig.height=8}
summary(mushroom)
str(mushroom)

mushroom%>%
  gather(key='key',value='value')%>%
  ggplot()+
  geom_histogram(mapping = aes(x=value,fill=key),stat="count")+
  #scale='free': they can have different scales on both x and y
  facet_wrap(~key,scales = 'free')+
  theme_minimal()
```

After exploring all variables through graphic method, we intend to drop five variables with low information value. They are gill_attachment,gill_spacing,ring_number,veil_color,veil_type. To confirm their exact distributions, we further use statistical method to confirm our decision. 

### Statistical Method 
```{r,message=FALSE}
library(Hmisc)
```

```{r}
mushroom%>%
  select(gill_attachment,gill_spacing,ring_number,veil_color,veil_type)%>%
  describe()
```

> Since these imbalanced features all include one level up to 80%, which we believe will lead to low information value problem, we are determined to drop these five variables.

```{r}
mushroom <- mushroom%>%
  select(-gill_attachment,-gill_spacing,-ring_number,-veil_color,-veil_type)

head(mushroom)
```

## 2. Split Data - Stratified Sampling
```{r}
library(caTools)
set.seed(1234)
mushroom_set <- mushroom%>%
  pull(edible)%>%
  sample.split(SplitRatio = 0.60)
mushroom_train <- subset(mushroom, mushroom_set==TRUE)
mushroom_test <- subset(mushroom, mushroom_set==FALSE)
```

Check the distribution:
```{r}
#origianl dataset
prop.table(table(mushroom$edible))

#trainset
prop.table(table(mushroom_train$edible))

#testset
prop.table(table(mushroom_test$edible))
```

# ● PART III: Train the Models 
Since all the features are nominal variables, we don't think it is good to use KNN model because it would be hard and meaningless to calculate the distances for nominal variables. Therefore, we will use decision tree model first and then use logistic regression.

## Build model

### Decision Tree
```{r}
library(rpart)

tree_mod <-
  rpart(
    edible ~ .,
    method = "class",
    data = mushroom_train,
    control = rpart.control(cp = 0.001)
  )

library(rpart.plot)
rpart.plot(tree_mod)
```

### Logistic Regression

>For logistic regression:\
1. We choose the *odor* feature in the first nodes of our decision tree as the main predictor in the logistic regression model, because the first nodes selected in the decision tree provide largest information gain with only one variable in the model.\
2. To confirm the first argument, we put every feature in the logistic model. And check its p-value.


run the odor variable first:
```{r}
logit_mod <- glm(edible ~ ., family=binomial(link="logit") , data = mushroom_train)
summary(logit_mod)
```



> After putting all variables as the predictor of edible in logistic regression model, we confirm that the *variable - odor* has the lowest AIC. Therefore, we are sure that if only one variable can be used in the logistic regression, the variable should be odor.

# ● PART IV: Evaluate the Performance of the Model

## 1. Predict 

### Decision Tree
```{r}
#probability prediction
tree_pred_prob <- predict(tree_mod, mushroom_test)
head(tree_pred_prob)

#classification prediction
tree_pred <- predict(tree_mod, mushroom_test,  type = "class")
head(tree_pred)
```

### Logistic Regression
```{r}
#probability prediction
logit_pred <- predict(logit_mod,  mushroom_test, type = 'response')
head(logit_pred)

#classification prediction
logit_pred <- ifelse(logit_pred > 0.5, 'Yes', 'No')
head(logit_pred)
```


## 2. Confusion Matrix

### Decision Tree
```{r}
tree_pred_table <- table(mushroom_test$edible, tree_pred)
tree_pred_table
```

### Logistic Regression
```{r}
logit_pred_table <- table(mushroom_test$edible, logit_pred)
logit_pred_table
```

## 3. Compare two models

### Accuracy for Decision Tree
```{r}
sum(diag(tree_pred_table)) / nrow(mushroom_test)
```

### Accuracy for Logistic Regression
```{r}
sum(diag(logit_pred_table)) / nrow(mushroom_test)
```


> Conclusion: in the cases above, we should choose decision tree for the two reasons listed below:\
1. Decision tree has a better accuracy. \
2. After looking at the confusion matrix, both of the two model has 0 number for mushroom which is actual edible but is predicted as not edible, but logistic regression model has a way higher number of mushrooms which is actual not edible but is predicted as edible (situation we want to prevent because of the terrible result).
