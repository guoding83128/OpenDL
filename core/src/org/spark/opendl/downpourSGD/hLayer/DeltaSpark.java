package org.spark.opendl.downpourSGD.hLayer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.spark.opendl.downpourSGD.SGDTrainConfig;
import org.spark.opendl.downpourSGD.SampleVector;
import org.spark.opendl.util.MathUtil;

import scala.Tuple2;
import spark.api.java.function.Function;

final class DeltaSpark extends Function<Tuple2<Integer, List<SampleVector>>, DeltaSpark> {
    private static final long serialVersionUID = 1L;
    private HiddenLayer hLayer;
    private SGDTrainConfig trainConfig;
    private DoubleMatrix my_w;
    private DoubleMatrix my_hbias;
    private DoubleMatrix my_vbias;
    private int curr_epoch = 0;

    public DeltaSpark(HiddenLayer _hlayer, SGDTrainConfig config, int epoch) {
        this.hLayer = _hlayer;
        this.trainConfig = config;
        this.curr_epoch = epoch;
        this.my_w = this.hLayer.getW().dup();
        this.my_hbias = this.hLayer.getHBias().dup();
        this.my_vbias = this.hLayer.getVBias().dup();
    }

    public DoubleMatrix getW() {
        return this.my_w;
    }
    public DoubleMatrix getHbias() {
        return this.my_hbias;
    }
    public DoubleMatrix getVbias() {
        return this.my_vbias;
    }
    public HiddenLayer getHLayer() {
        return this.hLayer;
    }

    @Override
    public DeltaSpark call(Tuple2<Integer, List<SampleVector>> arg) throws Exception {
        List<SampleVector> myList = new ArrayList<SampleVector>();
        for (SampleVector v: arg._2()) {
            myList.add(v);
        }
        Collections.shuffle(myList);
        
        DoubleMatrix x_samples = MathUtil.convertX2Matrix(myList);

        if (this.trainConfig.isUseCG() && (this.curr_epoch <= this.trainConfig.getCgEpochStep())) {
            this.hLayer.gradientUpdateCG(trainConfig, x_samples, my_w, my_hbias, my_vbias);
        } else {
        	this.hLayer.gradientUpdateMiniBatch(trainConfig, x_samples, my_w, my_hbias, my_vbias);
        }
        return this;
    }
}
