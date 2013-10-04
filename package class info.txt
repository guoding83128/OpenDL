OpenDL-The deep learning training library based on Spark framework

1 Package brief
  
1.1 package org.gd.spark.opendl.downpourSGD.hLayer
    The hidden layer data struct and train work architecture, meanly the core package. 

Class HiddenLayer: 
    Defines the abstract class for some possible hidden layer implementation(eg, AutoEncoder, dA, RBM). It has also some interface method related to reconstruct data and hidden layer output(sigmod). 
    
Class HiddenLayerTrain: 
    Define the static interface to hidden layer train work, both for multiple thread with standalone machine or Spark framework. 
    
Class HiddenLayerOptimizer: 
    Define the ConjugateGradient implementation for hidden layer. The sub class need just override the loss function and gradient update calculation implementation. Refer to Mallet project(http://mallet.cs.umass.edu/). 

Class DeltaThread:
    The model replica train thread for multiple thread standalone mechanism. The model replica idea refer to Google's DistBelief.

Class DeltaSpark:
    The model replca train function class for Spark framework. The model replica idea refer to Google's DistBelief.

Class LossThread:
    The loss calculation thread work.

Class LossSpark:
    The loss calculation spark work.

1.2 package org.gd.spark.opendl.downpourSGD.hLayer.dA
    The Denoising Autoencoders implementation package, refer to http://deeplearning.net/tutorial/dA.html.

CLass dA:
    The sub-class of HiddenLayer, override the reconstruct, gradientUpdateMiniBatch(batch gradient descent), gradientUpdateCG(conjugate gradient); and also define the optimizer for dA algorithm. 

1.3 package org.gd.spark.opendl.downpourSGD.hLayer.RBM
    The Restricted Boltzmann Machines implementation package, refer to http://deeplearning.net/tutorial/rbm.html.

Class RBM:
    The sub-class of HiddenLayer, override the reconstruct, gradientUpdateMiniBatch(batch gradient descent), gradientUpdateCG(conjugate gradient); and also define the optimizer for RBM algorithm.

1.4 package org.gd.spark.opendl.downpourSGD.lr
    The LogisticRegression(Softmax) related implementation package. In fact, there also many LR implementation utility package. In our LR work, still the downpourSGD mechanism for large scale data. 

Class LR:
    Define LogisticRegression(Softmax) node data struct, also some interface for prediction.

Class LRTrain:
    LR train work both for multiple thread work or Spark framework.

Class DeltaThread, DeltaSpark, LossThread, LossSpark:
    Similar with HiddenLayer package class with same name. 

1.5 package org.gd.spark.opendl.downpourSGD
    The downpourSGD mechanism related package.

Class ModelReplicaSplit:
    Define the data split for model replica with Spark framework. 

Class SampleVector:
    Base data struct of sample both used for supervised or unsupervised learning.

Interface SGDPersistable:
    Define the deep networking node persist interface. 

Class SGDTrainConfig:
    Define all train parameter in one data struct. 

1.6 package org.gd.spark.opendl.util
    Contain some math utility class.

2 Example usage.

  Refer to example code and doc.  























 