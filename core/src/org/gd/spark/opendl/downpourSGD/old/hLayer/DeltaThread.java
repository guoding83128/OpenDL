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
package org.gd.spark.opendl.downpourSGD.old.hLayer;

import java.util.Collections;
import java.util.List;

import org.apache.log4j.Logger;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.util.MathUtil;
import org.jblas.DoubleMatrix;

final class DeltaThread implements Runnable {
    private static final Logger logger = Logger.getLogger(DeltaThread.class);
    private HiddenLayer hLayer;
    private SGDTrainConfig trainConfig;
    private DoubleMatrix my_w;
    private DoubleMatrix my_hbias;
    private DoubleMatrix my_vbias;
    private DoubleMatrix samples;
    private boolean running = false;
    private int curr_epoch = 0;

    protected DeltaThread(HiddenLayer hlayer, SGDTrainConfig config) {
        this.hLayer = hlayer;
        this.trainConfig = config;
    }

    protected DoubleMatrix getW() {
        return this.my_w;
    }

    protected DoubleMatrix getHbias() {
        return this.my_hbias;
    }

    protected DoubleMatrix getVbias() {
        return this.my_vbias;
    }

    protected DoubleMatrix getSamples() {
        return this.samples;
    }

    protected boolean isRunning() {
        return this.running;
    }

    protected void train(List<SampleVector> x, int epoch) {
        this.curr_epoch = epoch;
        Collections.shuffle(x);
        this.samples = MathUtil.convertX2Matrix(x);
        new Thread(this).start();
    }

    @Override
    public void run() {
        this.running = true;
        try {
            // always get latest param
            this.my_w = this.hLayer.getW().dup();
            this.my_hbias = this.hLayer.getHBias().dup();
            this.my_vbias = this.hLayer.getVBias().dup();

            if (this.trainConfig.isUseCG() && (this.curr_epoch <= this.trainConfig.getCgEpochStep())) {
                this.hLayer.gradientUpdateCG(this.trainConfig, this.samples, this.my_w, this.my_hbias, this.my_vbias);
            } else {
                this.hLayer.gradientUpdateMiniBatch(this.trainConfig, this.samples, this.my_w, this.my_hbias, this.my_vbias);
            }
        } catch (Throwable e) {
            logger.error("", e);
        }
        this.running = false;
    }
}
