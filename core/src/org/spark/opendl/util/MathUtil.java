package org.spark.opendl.util;

import java.util.List;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.spark.opendl.downpourSGD.SampleVector;

public final class MathUtil {
    private static Random rand = new Random(System.currentTimeMillis());

    public static double sigmod(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double uniform(double min, double max) {
        return rand.nextDouble() * (max - min) + min;
    }

    public static void sigmod(DoubleMatrix m) {
        MatrixFunctions.expi(m.negi()).addi(1.0).rdivi(1.0);
    }

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

    public static double vectorL2Diff(double[] v1, double[] v2, int feature) {
        double ret = 0;
        for (int i = 0; i < feature; i++) {
            double a1 = v1[i];
            double a2 = v2[i];
            ret += (a1 - a2) * (a1 - a2);
        }
        return Math.sqrt(ret);
    }

    public static void doubleArrayCopy(double[][] src, double[][] dest, int row, int col) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                dest[i][j] = src[i][j];
            }
        }
    }

    public static void doubleArrayCopy(double[] src, double[] dest, int size) {
        for (int i = 0; i < size; i++) {
            dest[i] = src[i];
        }
    }

    public static boolean comparePredict(double[] y, double[] predict_y) {
        double max = 0;
        int idx = -1;
        for (int i = 0; i < predict_y.length; i++) {
            if (predict_y[i] > max) {
                max = predict_y[i];
                idx = i;
            }
        }
        return (1 == y[idx]);
    }

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

    public static void main(String[] args) {

    }
}
