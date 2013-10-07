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
package org.gd.spark.opendl.downpourSGD.hLayer;

import org.apache.log4j.Logger;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.train.SGDParam;
import org.gd.spark.opendl.util.MathUtil;
import org.gd.spark.opendl.util.MyConjugateGradient;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Restricted Boltzmann Machines implementation <p/>
 * refer to http://deeplearning.net/tutorial/rbm.html
 * 
 * @author GuoDing
 * @since 2013-08-15
 */
public class RBM extends HiddenLayer {
    private static final Logger logger = Logger.getLogger(RBM.class);
    private static final long serialVersionUID = 1L;

    public RBM(int in_n_visible, int in_n_hidden) {
        super(in_n_visible, in_n_hidden);
    }

    public RBM(int in_n_visible, int in_n_hidden, double[][] _w, double[] _b) {
        super(in_n_visible, in_n_hidden,_w, _b);
    }

    private void sample_h_given_v(DoubleMatrix v0_sample, DoubleMatrix mean, DoubleMatrix sample, DoubleMatrix curr_w,
            DoubleMatrix curr_hbias) {
        mean.copy(v0_sample.mmul(curr_w.transpose()).addiRowVector(curr_hbias));
        MathUtil.sigmod(mean);
        for (int i = 0; i < mean.rows; i++) {
            for (int j = 0; j < mean.columns; j++) {
                sample.put(i, j, MathUtil.binomial(1, mean.get(i, j)));
            }
        }
    }

    private void sample_v_given_h(DoubleMatrix h0_sample, DoubleMatrix mean, DoubleMatrix sample, DoubleMatrix curr_w,
            DoubleMatrix curr_vbias) {
        mean.copy(h0_sample.mmul(curr_w).addiRowVector(curr_vbias));
        MathUtil.sigmod(mean);
        for (int i = 0; i < mean.rows; i++) {
            for (int j = 0; j < mean.columns; j++) {
                sample.put(i, j, MathUtil.binomial(1, mean.get(i, j)));
            }
        }
    }

    @Override
    protected void gradientUpdateMiniBatch(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param) {
    	int nbr_sample = x_samples.getRows();
    	DoubleMatrix curr_w = ((HiddenLayerParam)curr_param).w;
    	DoubleMatrix curr_hbias = ((HiddenLayerParam)curr_param).hbias;
    	DoubleMatrix curr_vbias = ((HiddenLayerParam)curr_param).vbias;
    	
    	DoubleMatrix ph_mean = new DoubleMatrix(nbr_sample, n_hidden);
    	DoubleMatrix ph_sample = new DoubleMatrix(nbr_sample, n_hidden);
    	DoubleMatrix nv_means = new DoubleMatrix(nbr_sample, n_visible);
    	DoubleMatrix nv_samples = new DoubleMatrix(nbr_sample, n_visible);
    	DoubleMatrix nh_means = new DoubleMatrix(nbr_sample, n_hidden);
    	DoubleMatrix nh_samples = new DoubleMatrix(nbr_sample, n_hidden);
    	
    	sample_h_given_v(x_samples, ph_mean, ph_sample, curr_w, curr_hbias);
        sample_v_given_h(ph_sample, nv_means, nv_samples, curr_w, curr_vbias);
        sample_h_given_v(nv_samples, nh_means, nh_samples, curr_w, curr_hbias);
        
        DoubleMatrix delta_w = ph_mean.transpose().mmul(x_samples).subi(nh_means.transpose().mmul(nv_samples)).divi(nbr_sample);
        DoubleMatrix delta_hbias = ph_sample.sub(nh_means).columnSums().divi(nbr_sample);
        DoubleMatrix delta_vbias = x_samples.sub(nv_samples).columnSums().divi(nbr_sample);
        
        curr_w.addi(delta_w.muli(config.getLearningRate()));
        curr_hbias.addi(delta_hbias.transpose().muli(config.getLearningRate()));
        curr_vbias.addi(delta_vbias.transpose().muli(config.getLearningRate()));
    }

    @Override
    protected void gradientUpdateCG(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param) {
    	DoubleMatrix curr_w = ((HiddenLayerParam)curr_param).w;
    	DoubleMatrix curr_hbias = ((HiddenLayerParam)curr_param).hbias;
    	DoubleMatrix curr_vbias = ((HiddenLayerParam)curr_param).vbias;
    	
    	RBMOptimizer rbmopt = new RBMOptimizer(config, x_samples, n_visible, n_hidden, curr_w, curr_hbias, curr_vbias);
        MyConjugateGradient cg = new MyConjugateGradient(rbmopt, config.getCgInitStepSize());
        cg.setTolerance(config.getCgTolerance());
        try {
            cg.optimize(config.getCgMaxIterations());
        } catch (Throwable e) {
            logger.error("", e);
        }
    }

    @Override
    public DoubleMatrix reconstruct(DoubleMatrix input) {
        DoubleMatrix ret = input.mmul(hlparam.w.transpose()).addiRowVector(hlparam.hbias);
        MathUtil.sigmod(ret);
        ret = ret.mmul(hlparam.w).addiRowVector(hlparam.vbias);
        MathUtil.sigmod(ret);
        return ret;
    }
    
	@Override
	public void reconstruct(double[] x, double[] reconstruct_x) {
		DoubleMatrix x_m = new DoubleMatrix(x).transpose();
    	DoubleMatrix ret = reconstruct(x_m);
    	for(int i = 0; i < n_visible; i++) {
    		reconstruct_x[i] = ret.get(0, i);
    	}
	}

    private class RBMOptimizer extends HiddenLayerOptimizer {
        private DoubleMatrix ph_mean;
        private DoubleMatrix ph_sample;
        private DoubleMatrix nv_means;
        private DoubleMatrix nv_samples;
        private DoubleMatrix nh_means;
        private DoubleMatrix nh_samples;

        public RBMOptimizer(SGDTrainConfig config, DoubleMatrix samples, int n_visible, int n_hidden, DoubleMatrix w,
                DoubleMatrix hbias, DoubleMatrix vbias) {
            super(config, samples, n_visible, n_hidden, w, hbias, vbias);
            ph_mean = new DoubleMatrix(nbr_sample, n_hidden);
            ph_sample = new DoubleMatrix(nbr_sample, n_hidden);
            nv_means = new DoubleMatrix(nbr_sample, n_visible);
            nv_samples = new DoubleMatrix(nbr_sample, n_visible);
            nh_means = new DoubleMatrix(nbr_sample, n_hidden);
            nh_samples = new DoubleMatrix(nbr_sample, n_hidden);
        }

        @Override
        public double getValue() {
            sample_h_given_v(my_samples, ph_mean, ph_sample, my_w, my_hbias);
            sample_v_given_h(ph_sample, nv_means, nv_samples, my_w, my_vbias);
            double loss = MatrixFunctions.powi(my_samples.sub(nv_means), 2).sum() / nbr_sample;
            return -loss;
        }

        @Override
        public void getValueGradient(double[] arg) {
            sample_h_given_v(nv_samples, nh_means, nh_samples, my_w, my_hbias);
            DoubleMatrix delta_w = ph_mean.transpose().mmul(my_samples).subi(nh_means.transpose().mmul(nv_samples)).divi(nbr_sample);
            DoubleMatrix delta_hbias = ph_sample.sub(nh_means).columnSums().divi(nbr_sample);
            DoubleMatrix delta_vbias = my_samples.sub(nv_samples).columnSums().divi(nbr_sample);

            int idx = 0;
            for (int i = 0; i < n_hidden; i++) {
                for (int j = 0; j < n_visible; j++) {
                    arg[idx++] = delta_w.get(i, j);
                }
            }
            for (int i = 0; i < n_hidden; i++) {
                arg[idx++] = delta_hbias.get(0, i);
            }
            for (int i = 0; i < n_visible; i++) {
                arg[idx++] = delta_vbias.get(0, i);
            }
        }
    }
}
