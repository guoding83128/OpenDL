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
