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
