package org.gd.spark.opendl.downpourSGD.old.hLayer;

import java.util.List;

import org.apache.spark.api.java.function.Function;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.util.MathUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import scala.Tuple2;

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
