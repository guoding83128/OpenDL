/**
 * @(#)ModelReplicaSplit.java, 2013-8-15. 
 * 
 * Copyright 2013 NetEase, Inc. All rights reserved.
 * NetEase PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */
package org.spark.opendl.downpourSGD;

import java.io.Serializable;
import java.util.List;
import java.util.Random;

import scala.Tuple2;
import spark.api.java.JavaPairRDD;
import spark.api.java.JavaRDD;
import spark.api.java.function.PairFunction;

public final class ModelReplicaSplit<T> implements Serializable {
    private static final long serialVersionUID = 1L;
    private Random rand = new Random(System.currentTimeMillis());

    public JavaPairRDD<Integer, List<T>> split(JavaRDD<T> input, int nrModelReplica, boolean cache) {
        JavaPairRDD<Integer, List<T>> output = input.map(new SplitModelReplica(nrModelReplica)).groupByKey();
        if (cache) {
            return output.cache();
        }
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
