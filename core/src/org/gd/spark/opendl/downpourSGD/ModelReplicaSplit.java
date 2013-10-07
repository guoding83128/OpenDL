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
package org.gd.spark.opendl.downpourSGD;

import java.io.Serializable;
import java.util.List;
import java.util.Random;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

/**
 * Samples split for Spark train work <p/>
 * 
 * @author GuoDing
 * @since 2013-07-20
 * @param <T>
 */
public final class ModelReplicaSplit<T> implements Serializable {
    private static final long serialVersionUID = 1L;
    private Random rand = new Random(System.currentTimeMillis());

    /**
     * Split the input samples (one each split for one ModelReplica)
     * 
     * @param input
     * @param nrModelReplica
     * @param cache
     * @return
     */
    public JavaPairRDD<Integer, List<T>> split(JavaRDD<T> input, int nrModelReplica, SGDTrainConfig config) {
        JavaPairRDD<Integer, List<T>> output = input.map(new SplitModelReplica(nrModelReplica)).groupByKey().persist(config.getMrDataStorage());
        output.count();
        return output;
    }

    private class SplitModelReplica extends PairFunction<T, Integer, T> {
        private static final long serialVersionUID = 1L;
        private int nrModelReplica;

        public SplitModelReplica(int nr) {
            this.nrModelReplica = nr;
        }

        @Override
        public Tuple2<Integer, T> call(T arg) throws Exception {
            int idx = rand.nextInt(nrModelReplica);
            return new Tuple2<Integer, T>(idx, arg);
        }
    }
}
