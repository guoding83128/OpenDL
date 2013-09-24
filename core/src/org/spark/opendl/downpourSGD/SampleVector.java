package org.spark.opendl.downpourSGD;

import java.io.Serializable;

public class SampleVector implements Serializable {
    private static final long serialVersionUID = 1L;
    protected double[] x;
    protected double[] y;

    public SampleVector(int x_feature) {
        this(x_feature, 0);
    }

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
