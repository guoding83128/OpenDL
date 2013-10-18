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
package org.gd.spark.opendl.downpourSGD.TiedWeightLayer;

import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.jblas.DoubleMatrix;

import cc.mallet.optimize.Optimizable;

/**
 * ConjugateGradient optimizer implementation for hidden layer(dA, RBM) <p/>
 * 
 * @author GuoDing
 * @since 2013-08-15
 */
abstract class HiddenLayerOptimizer implements Optimizable.ByGradientValue {
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
