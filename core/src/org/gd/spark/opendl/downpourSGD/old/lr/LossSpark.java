/*
 * Copyright 2013 GuoDing
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.gd.spark.opendl.downpourSGD.old.lr;

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
