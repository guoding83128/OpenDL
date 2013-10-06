package org.gd.spark.opendl.downpourSGD.train;

import java.util.List;

import scala.Tuple2;

import org.apache.spark.api.java.function.Function;
import org.gd.spark.opendl.downpourSGD.SampleVector;

final class LossSpark extends Function<Tuple2<Integer, List<SampleVector>>, Double> {
    private static final long serialVersionUID = 1L;
    private SGDBase sgd;

    public LossSpark(SGDBase _sgd) {
        this.sgd = _sgd;
    }

    @Override
    public Double call(Tuple2<Integer, List<SampleVector>> arg) throws Exception {
    	return this.sgd.loss(arg._2());
    }
}
