package org.gd.spark.opendl.downpourSGD.lr;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import scala.Tuple2;

import org.apache.spark.api.java.function.Function;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.util.MathUtil;
import org.jblas.DoubleMatrix;

final class DeltaSpark extends Function<Tuple2<Integer, List<SampleVector>>, DeltaSpark> {
    private static final long serialVersionUID = 1L;
    private LR lr;
    private SGDTrainConfig trainConfig;
    private DoubleMatrix my_w;
    private DoubleMatrix my_b;
    private int curr_epoch = 0;

    public DeltaSpark(LR _lr, SGDTrainConfig config, int epoch) {
        this.lr = _lr;
        this.trainConfig = config;
        this.curr_epoch = epoch;
        my_w = lr.getW().dup();
        my_b = lr.getB().dup();
    }

    public DoubleMatrix getW() {
        return this.my_w;
    }

    public DoubleMatrix getB() {
        return this.my_b;
    }

    public LR getLR() {
        return this.lr;
    }

    @Override
    public DeltaSpark call(Tuple2<Integer, List<SampleVector>> arg) throws Exception {
        List<SampleVector> myList = new ArrayList<SampleVector>();
        for (SampleVector v: arg._2()) {
            myList.add(v);
        }
        Collections.shuffle(myList);

        DoubleMatrix x_samples = MathUtil.convertX2Matrix(myList);
        DoubleMatrix y_samples = MathUtil.convertY2Matrix(myList);

        // check whether we use cg this time
        if (this.trainConfig.isUseCG() && (this.curr_epoch <= this.trainConfig.getCgEpochStep())) {
            this.lr.gradientUpdateCG(trainConfig, x_samples, y_samples, my_w, my_b);
        } else {
            this.lr.gradientUpdateMiniBatch(trainConfig, x_samples, y_samples, my_w, my_b);
        }
        return this;
    }
}
