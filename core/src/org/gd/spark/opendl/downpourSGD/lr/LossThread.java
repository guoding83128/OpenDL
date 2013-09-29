package org.gd.spark.opendl.downpourSGD.lr;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

final class LossThread implements Runnable {
    private LR lr;
    private DoubleMatrix x_samples;
    private DoubleMatrix y_samples;
    private boolean running = false;
    private double error = 0;

    public LossThread(LR _lr) {
        this.lr = _lr;
    }

    public double getError() {
        return this.error;
    }

    public boolean isRunning() {
        return this.running;
    }

    public void sumLoss(DoubleMatrix x, DoubleMatrix y) {
        this.x_samples = x;
        this.y_samples = y;
        this.error = 0;
        new Thread(this).start();
    }

    @Override
    public void run() {
        this.running = true;
        DoubleMatrix predict_y = this.lr.predict(x_samples);
        this.error = MatrixFunctions.powi(predict_y.sub(y_samples), 2).sum();
        this.running = false;
    }
}
