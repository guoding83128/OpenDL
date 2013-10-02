package org.gd.spark.opendl.downpourSGD.lr;

import java.util.List;

import scala.Tuple2;

import org.apache.spark.api.java.function.Function;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.util.MathUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

final class LossSpark extends Function<Tuple2<Integer, List<SampleVector>>, Double> {
    private static final long serialVersionUID = 1L;
    private LR lr;

    public LossSpark(LR _lr) {
        this.lr = _lr;
    }

    @Override
    public Double call(Tuple2<Integer, List<SampleVector>> arg) throws Exception {
        DoubleMatrix x_samples = MathUtil.convertX2Matrix(arg._2());
        DoubleMatrix y_samples = MathUtil.convertY2Matrix(arg._2());
        DoubleMatrix predict_y = this.lr.predict(x_samples);
        return MatrixFunctions.powi(predict_y.sub(y_samples), 2).sum();
    }
}
