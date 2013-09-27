package org.spark.opendl.downpourSGD.hLayer;

import java.util.List;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.spark.opendl.downpourSGD.SampleVector;
import org.spark.opendl.util.MathUtil;

import scala.Tuple2;
import spark.api.java.function.Function;

final class LossSpark extends Function<Tuple2<Integer, List<SampleVector>>, Double> {
    private static final long serialVersionUID = 1L;
    private HiddenLayer hLayer;

    public LossSpark(HiddenLayer hlayer) {
        this.hLayer = hlayer;
    }

    @Override
    public Double call(Tuple2<Integer, List<SampleVector>> arg) throws Exception {
        DoubleMatrix x = MathUtil.convertX2Matrix(arg._2());
        DoubleMatrix reconstruct_x = this.hLayer.reconstruct(x);
        return MatrixFunctions.powi(reconstruct_x.sub(x), 2).sum();
    }
}
