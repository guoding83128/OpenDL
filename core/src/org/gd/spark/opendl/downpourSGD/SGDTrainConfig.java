/*
 * Copyright 2013 GuoDing
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.gd.spark.opendl.downpourSGD;

import java.io.Serializable;

import org.apache.spark.storage.StorageLevel;

import lombok.Data;

/**
 * DownpourSGD train work configuration parameter <p/>
 * 
 * @author GuoDing
 * @since 2013-07-15
 */
@Data
public class SGDTrainConfig implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * number of model replica
     */
    public int nbrModelReplica = 1;
    
    /**
     * specify the spark persistent level for split train sample data
     */
    public StorageLevel mrDataStorage = StorageLevel.MEMORY_ONLY();

    /**
     * stop condition
     */
    public int maxEpochs = 500;
    public double minLoss = 0.1;

    /**
     * learning rate
     */
    public double learningRate = 0.001; // only for fix lr, or adagrad basic lr

    /**
     * regulation
     */
    public boolean useRegularization = false;
    public double lamada1 = 0; // most case, only use L2 reg is ok
    public double lamada2 = 0.0001;

    /**
     * caculate loss function
     */
    public int lossCalStep = 1; //calculate and print total loss every calStep
    public boolean printLoss = false;
    
    /**
     * parameter output during each epoche of training
     */
    public boolean paramOutput = false;
    public int paramOutputStep = 5; //parameter output every paramOutputStep
    public String paramOutputPath = null;

    /**
     * for dA
     */
    public boolean doCorruption = true;
    public double corruption_level = 0.3;

    /**
     * use ConjugateGradient on front step
     */
    // whether use cg on front step
    public boolean useCG = false;

    // stop use cg(switch to asynchronous SGD) after cgEpochStep epoch times.
    // default is 1, means only use cg first time(coverage fast to low loss).
    public int cgEpochStep = 1;

    // max iteration time in one time cg
    public int cgMaxIterations = 100;

    // the cg min gradient update tolerance
    public double cgTolerance = 0.0001;

    // cg init step size
    public double cgInitStepSize = 0.01;
}
