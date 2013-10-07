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

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

final class LossThread implements Runnable {
    private HiddenLayer hLayer;
    private DoubleMatrix samples;
    private boolean running = false;
    private double error = 0;

    protected LossThread(HiddenLayer hlayer) {
        this.hLayer = hlayer;
    }

    protected double getError() {
        return this.error;
    }

    protected boolean isRunning() {
        return this.running;
    }

    protected void sumLoss(DoubleMatrix x) {
        this.samples = x;
        this.error = 0;
        new Thread(this).start();
    }

    @Override
    public void run() {
        this.running = true;
        DoubleMatrix z = hLayer.reconstruct(samples);
        this.error = MatrixFunctions.pow(z.subi(samples), 2).sum();
        this.running = false;
    }
}
