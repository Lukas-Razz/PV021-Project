# PV021-Project

```
mvn clean install
cd target
java -jar NeuralNets-1.0-SNAPSHOT-jar-with-dependencies.jar learningRate=0.001 momentum=0.03 l1=0.00001 l2=0.002 batchSize=1 epochs=80 loss=SE optim=SGD init=Normal sizes=[72,36] activ=[Softsign,Softsign]
```
Presented parameters are default values.
Parameters are space-separated.

Avaible functions:
  Optimization algorithms (optim):
    SDG - Stochastic Gradient Descend
    AdaGrad - Adaptive Gradient
    
  Loss functions (loss):
    SE - Squared Error
    MSE - Mean Squared Error
    RMSE - Root Mean Square Error
  
  Initialization (init):
    Uniform
    Normal
    
  Activativation Functions (activ):
    Softsign
    Tanh
    Sigmoid
    
Arrays (sizes and activ) must have the same size.
