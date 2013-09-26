package org.spark.opendl.downpourSGD;

import java.io.Serializable;

/**
 * Sample can contains supervised data or unsupervised data<p/>
 * 
 * @author GuoDing
 * @since 2013-07-23
 */
public class SampleVector implements Serializable {
    private static final long serialVersionUID = 1L;
    private double[] x;
    private double[] y;

    /**
     * Constructor with unsupervised data
     * 
     * @param x_feature Feature number
     */
    public SampleVector(int x_feature) {
        this(x_feature, 0);
    }

    /**
     * Constructor with supervised data
     * 
     * @param x_feature Feature number
     * @param y_feature Class number
     */
    public SampleVector(int x_feature, int y_feature) {
        x = new double[x_feature];
        y = new double[y_feature];
    }

    public SampleVector(double[] _x, double[] _y) {
        x = _x;
        y = _y;
    }

    public boolean isSupervise() {
        return (0 == y.length);
    }

    public double[] getX() {
        return x;
    }

    public double[] getY() {
        return y;
    }

    public void setX(double[] _x) {
        x = _x;
    }

    public void setY(double[] _y) {
        y = _y;
    }

    public String toString() {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < x.length; i++) {
            sb.append(x[i]);
            sb.append(",");
        }
        return sb.toString();
    }
}
