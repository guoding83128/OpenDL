/**
 * @(#)HiddenLayerOptimizer.java, 2013-8-28. 
 * 
 * Copyright 2013 NetEase, Inc. All rights reserved.
 * NetEase PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */
package org.spark.opendl.downpourSGD.hLayer;

import org.jblas.DoubleMatrix;
import org.spark.opendl.downpourSGD.SGDTrainConfig;

import cc.mallet.optimize.Optimizable;

public abstract class HiddenLayerOptimizer implements Optimizable.ByGradientValue {
    protected SGDTrainConfig myConfig;
    protected int my_n_visible;
    protected int my_n_hidden;
    protected int nbr_sample;
    protected DoubleMatrix my_w;
    protected DoubleMatrix my_hbias;
    protected DoubleMatrix my_vbias;
    protected DoubleMatrix my_samples;

    public HiddenLayerOptimizer(SGDTrainConfig config, DoubleMatrix samples, int n_visible, int n_hidden,
            DoubleMatrix w, DoubleMatrix hbias, DoubleMatrix vbias) {
        myConfig = config;
        my_samples = samples;
        nbr_sample = samples.getRows();
        my_n_visible = n_visible;
        my_n_hidden = n_hidden;
        my_w = w;
        my_hbias = hbias;
        my_vbias = vbias;
    }

    @Override
    public final int getNumParameters() {
        return my_n_hidden * my_n_visible + my_n_hidden + my_n_visible;
    }

    @Override
    public final double getParameter(int arg) {
        if (arg < my_n_hidden * my_n_visible) {
            int i = arg / my_n_visible;
            int j = arg % my_n_visible;
            return my_w.get(i, j);
        } else if (arg < my_n_hidden * my_n_visible + my_n_hidden) {
            return my_hbias.get(arg - my_n_hidden * my_n_visible, 0);
        }
        return my_vbias.get(arg - my_n_hidden * my_n_visible - my_n_hidden, 0);
    }

    @Override
    public final void getParameters(double[] arg) {
        int idx = 0;
        for (int i = 0; i < my_n_hidden; i++) {
            for (int j = 0; j < my_n_visible; j++) {
                arg[idx++] = my_w.get(i, j);
            }
        }
        for (int i = 0; i < my_n_hidden; i++) {
            arg[idx++] = my_hbias.get(i, 0);
        }
        for (int i = 0; i < my_n_visible; i++) {
            arg[idx++] = my_vbias.get(i, 0);
        }
    }

    @Override
    public final void setParameter(int arg0, double arg1) {
        if (arg0 < my_n_hidden * my_n_visible) {
            int i = arg0 / my_n_visible;
            int j = arg0 % my_n_visible;
            my_w.put(i, j, arg1);
        } else if (arg0 < my_n_hidden * my_n_visible + my_n_hidden) {
            my_hbias.put(arg0 - my_n_hidden * my_n_visible, 0, arg1);
        } else {
            my_vbias.put(arg0 - my_n_hidden * my_n_visible - my_n_hidden, 0, arg1);
        }
    }

    @Override
    public final void setParameters(double[] arg) {
        int idx = 0;
        for (int i = 0; i < my_n_hidden; i++) {
            for (int j = 0; j < my_n_visible; j++) {
                my_w.put(i, j, arg[idx++]);
            }
        }
        for (int i = 0; i < my_n_hidden; i++) {
            my_hbias.put(i, 0, arg[idx++]);
        }
        for (int i = 0; i < my_n_visible; i++) {
            my_vbias.put(i, 0, arg[idx++]);
        }
    }
}
