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
