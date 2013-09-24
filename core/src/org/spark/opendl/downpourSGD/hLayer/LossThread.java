package org.spark.opendl.downpourSGD.hLayer;

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
