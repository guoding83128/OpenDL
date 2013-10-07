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
package org.gd.spark.opendl.downpourSGD.old.lr;

import java.util.Collections;
import java.util.List;

import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.util.MathUtil;
import org.jblas.DoubleMatrix;

final class DeltaThread implements Runnable {
    private LR lr;
    private SGDTrainConfig trainConfig;
    private DoubleMatrix my_w;
    private DoubleMatrix my_b;
    private DoubleMatrix x_samples;
    private DoubleMatrix y_samples;
    private boolean running = false;
    private int curr_epoch = 0;

    public DeltaThread(LR _lr, SGDTrainConfig config, List<SampleVector> xy) {
        this.lr = _lr;
        this.trainConfig = config;
        Collections.shuffle(xy);
        this.x_samples = MathUtil.convertX2Matrix(xy);
        this.y_samples = MathUtil.convertY2Matrix(xy);
    }

    public DoubleMatrix getW() {
        return this.my_w;
    }

    public DoubleMatrix getB() {
        return this.my_b;
    }

    public DoubleMatrix getX() {
        return this.x_samples;
    }

    public DoubleMatrix getY() {
        return this.y_samples;
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
		this.my_w = lr.getW().dup();
		this.my_b = lr.getB().dup();

		// check whether we use cg this time
		if (this.trainConfig.isUseCG() && (this.curr_epoch <= this.trainConfig.getCgEpochStep())) {
			this.lr.gradientUpdateCG(trainConfig, x_samples, y_samples, my_w, my_b);
		} else {
			this.lr.gradientUpdateMiniBatch(trainConfig, x_samples, y_samples, my_w, my_b);
		}

		this.running = false;
	}
}
