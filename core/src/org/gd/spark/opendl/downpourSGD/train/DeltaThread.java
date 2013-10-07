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
package org.gd.spark.opendl.downpourSGD.train;

import java.util.Collections;
import java.util.List;

import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.util.MathUtil;
import org.jblas.DoubleMatrix;

final class DeltaThread implements Runnable {
    private SGDBase sgd;
    private SGDTrainConfig trainConfig;
    private SGDParam my_param;
    private List<SampleVector> samples;
    private DoubleMatrix x_samples;
    private DoubleMatrix y_samples;
    private boolean running = false;
    private int curr_epoch = 0;

    public DeltaThread(SGDBase _sgd, SGDTrainConfig config, List<SampleVector> xy) {
        this.sgd = _sgd;
        this.trainConfig = config;
        Collections.shuffle(xy);
        this.samples = xy;
        this.x_samples = MathUtil.convertX2Matrix(xy);
        if(this.sgd.isSupervise()) {
            this.y_samples = MathUtil.convertY2Matrix(xy);
        }
    }
    
    public SGDParam getParam() {
    	return this.my_param;
    }
    
    public List<SampleVector> getSamples() {
    	return this.samples;
    }

    public boolean isRunning() {
        return this.running;
    }

    public void train(int epoch) {
        this.curr_epoch = epoch;
        new Thread(this).start();
    }

    @Override
	public void run() {
		this.running = true;
		
		// always get latest param
		this.my_param = this.sgd.getParam().dup();

		// check whether we use cg this time
		if (this.trainConfig.isUseCG() && (this.curr_epoch <= this.trainConfig.getCgEpochStep())) {
			this.sgd.gradientUpdateCG(trainConfig, x_samples, y_samples, my_param);
		} else {
			this.sgd.gradientUpdateMiniBatch(trainConfig, x_samples, y_samples, my_param);
		}

		this.running = false;
	}
}
