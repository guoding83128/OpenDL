package org.spark.opendl.downpourSGD.lr;

import java.util.Collections;
import java.util.List;

import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.spark.opendl.downpourSGD.SGDTrainConfig;
import org.spark.opendl.downpourSGD.SampleVector;
import org.spark.opendl.util.MathUtil;

final class DeltaThread implements Runnable {
    private static final Logger logger = Logger.getLogger(DeltaThread.class);
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
        try {
            // always get latest param
            this.my_w = lr.getW().dup();
            this.my_b = lr.getB().dup();

            // check whether we use cg this time
            if (this.trainConfig.isUseCG() && (this.curr_epoch <= this.trainConfig.getCgEpochStep())) {
                this.lr.gradientUpdateCG(trainConfig, x_samples, y_samples, my_w, my_b);
            } else {
                this.lr.gradientUpdateMiniBatch(trainConfig, x_samples, y_samples, my_w, my_b);
            }
        } catch (Throwable e) {
            logger.error("", e);
        }
        this.running = false;
    }
}
