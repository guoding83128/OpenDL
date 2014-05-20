OpenDL-The deep learning training library based on Spark framework

# 1 Core idea

The Google scientist, Jeffrey Dean promotes one way to large scale data��s DeepLearning training with distributed platform, named DistBelief [1]. The key idea is model replica, each one takes the same current model parameters, but get the different data shards to train; then each model replica update the gradient to central parameter server.

My framework splits the train data into different data shards, each one will be trained by the model replica. After all model replica finish the current epoch train, the update gradient will be reduced to update totally; then each model replica will start the next epoch train with new parameter until convergence or get to some stop conditions. The model replica can train the data with different way based on gradient update; eg, mini-batch gradient descent, Conjugate gradient, or L-BFGS.(CG always win the best result).

So the algorithm in OpenDL should be gradient update support. Like LogisticRegression(Softmax), Backpropagation, AutoEncoder, RBM, Convolution and so on, all of them can be incorporated into OpenDL framework. 

# 2 Third party software

Besides some of the Apache common software modules, the OpenDL developed with three open source project.
    
The Spark light cluster computing platform, refer to http://spark.incubator.apache.org/. Now we just use the latest version, 0.8.0 just released recently. 
    
The Mallet, java based machine learning package of UMASS, refer to http://mallet.cs.umass.edu/. We use this one mainly for mathematical algorithm, eg, conjugate gradient, L-BFGS. 
    
The last is JBlas, library of Linear Algebra for Java, refer to http://mikiobraun.github.io/jblas/. It has been used mainly for matrix computation optimization. So before we run the OpenDL program on Windows, Unix, Linux, MacOS, must check the OS platform for JBlas runtime support with "java -server -jar jblas-1.2.3.jar", then get the basic install and benchmark information. 

# 3 File organization

    core/  Main core source and maven pom.xml.

    examples/ Some example code. 

    dep_lib/ All dependency jar file.

    dist/ Distribute jar.

    doc/ Include java doc, some related papers.

    Readme.txt Some information about the core source architecture. 


# 4 Content information
    Author info:
    GuoDing, 
    email: guoding83128@163.com, guoding83128@gmail.com
    Google group:
    broad will be created soon! 

# Any suggestion are welcome. 

# References
[1] Large Scale Distributed Deep Networks. Jeffrey Dean, Google Inc.
[2] Building High-level Features Using Large Scale Unsupervised Learning. Quoc V. Le,Marc'Aurelio Ranzato, Stanford & Google Inc.
