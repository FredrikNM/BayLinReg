# BayLinReg
A pymc3 version of Bayesian linear regression.
Pymc3 is a easy way to implement Baysian statistic. [Link to Pymc3's own examples](https://docs.pymc.io/nb_examples/index.html)
  
Quick statistic recap. We try to use a variables, or many, like the price of cheese and tomatos, to explain something else, like the price of pizza.  
Maybe it can be describe like this :  pizza_price = 2 * cheese_price + 0.5 * tomato_price. 
Or a trick you might have learned earlier,  
pizza_prices = cheese_price * cheese_price, or tomato_price^3. What ever it might be, if we are  
sure we have some variable that in some kind of way should describe something, we should try out some of these alternatives.  
Preferably we can do this automatically, with out all the hassle of thinking about all the alternatives we have.  
In this python code is a quick way of implementing this as well.  
  
Another thing addressed in the code is that tomato prices might just explain the cheaper variants of pizza, lets say a pizza  
between 5-10 bucks. But when you look at pizza's in the price range between 10-20 bucks, maybe we need to take a closer  
look at beef prices. The quick-fix for this in the code is a check for correlation in different scales of the variable we  
try to explain. This could be done in different ways as well, e. g. making hierarchical models. 
  

