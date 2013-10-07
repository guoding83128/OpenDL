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
package org.gd.spark.opendl.util;

import java.util.List;
import java.util.Random;

import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Some mathematic utility function <p/>
 * 
 * @author GuoDing
 * @since 2013-07-01
 */
public final class MathUtil {
    private static Random rand = new Random(System.currentTimeMillis());

    /**
     * Sigmod function
     * @param z
     * @return
     */
    public static double sigmod(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /**
     * Uniform with random in some range
     * @param min
     * @param max
     * @return
     */
    public static double uniform(double min, double max) {
        return rand.nextDouble() * (max - min) + min;
    }

    /**
     * Sigmod for each node in matrix
     * @param m
     */
    public static void sigmod(DoubleMatrix m) {
        MatrixFunctions.expi(m.negi()).addi(1.0).rdivi(1.0);
    }

    /**
     * Corruption process for dA
     * @param n
     * @param p
     * @return
     */
    public static int binomial(int n, double p) {
        if ((p < 0) || (p > 1)) {
            return 0;
        }
        int c = 0;
        for (int i = 0; i < n; i++) {
            if (rand.nextDouble() < p) {
                c++;
            }
        }
        return c;
    }

    /**
     * Convert sample X to matrix
     * @param samples
     * @return
     */
    public static DoubleMatrix convertX2Matrix(List<SampleVector> samples) {
        int row = samples.size();
        int col = samples.get(0).getX().length;

        DoubleMatrix ret = new DoubleMatrix(row, col);
        row = 0;
        for (SampleVector sample: samples) {
            double[] x = sample.getX();
            for (col = 0; col < x.length; col++) {
                ret.put(row, col, x[col]);
            }
            row++;
        }
        return ret;
    }

    /**
     * Convert class Y to matrix
     * @param samples
     * @return
     */
    public static DoubleMatrix convertY2Matrix(List<SampleVector> samples) {
        int row = samples.size();
        int col = samples.get(0).getY().length;

        DoubleMatrix ret = new DoubleMatrix(row, col);
        row = 0;
        for (SampleVector sample: samples) {
            double[] y = sample.getY();
            for (col = 0; col < y.length; col++) {
                ret.put(row, col, y[col]);
            }
            row++;
        }
        return ret;
    }
}
