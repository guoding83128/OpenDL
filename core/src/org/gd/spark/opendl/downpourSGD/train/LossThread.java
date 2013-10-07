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

import java.util.List;

import org.gd.spark.opendl.downpourSGD.SampleVector;

final class LossThread implements Runnable {
    private SGDBase sgd;
    private List<SampleVector> samples;
    private boolean running = false;
    private double error = 0;

    public LossThread(SGDBase _sgd) {
        this.sgd = _sgd;
    }

    public double getError() {
        return this.error;
    }

    public boolean isRunning() {
        return this.running;
    }

    public void sumLoss(List<SampleVector> xy) {
        this.samples = xy;
        this.error = 0;
        new Thread(this).start();
    }

    @Override
    public void run() {
        this.running = true;
        this.error = this.sgd.loss(this.samples);
        this.running = false;
    }
}
