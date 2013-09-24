package org.spark.opendl.downpourSGD.lr;

import java.util.List;

import scala.Tuple2;
import spark.api.java.function.Function;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.spark.opendl.downpourSGD.SampleVector;
import org.spark.opendl.util.MathUtil;

final class LossSpark extends Function<Tuple2<Integer, List<SampleVector>>, Double> {
    private static final long serialVersionUID = 1L;
    private LR lr;

    public LossSpark(LR _lr) {
        this.lr = _lr;
    }

    @Override
    public Double call(Tuple2<Integer, List<SampleVector>> arg) throws Exception {
        double error = 0;
        DoubleMatrix x_samples = MathUtil.convertX2Matrix(arg._2());
        DoubleMatrix y_samples = MathUtil.convertY2Matrix(arg._2());
        DoubleMatrix predict_y = this.lr.predict(x_samples);
        error = MatrixFunctions.powi(predict_y.sub(y_samples), 2).sum();
        return error;
    }
}
