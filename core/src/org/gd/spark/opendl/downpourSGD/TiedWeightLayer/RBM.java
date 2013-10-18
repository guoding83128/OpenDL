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

import org.apache.log4j.Logger;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.train.SGDParam;
import org.gd.spark.opendl.util.MathUtil;
import org.gd.spark.opendl.util.MyConjugateGradient;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Restricted Boltzmann Machines implementation <p/>
 * Fast Contrastive Divergence(CD1), Hinton 2002. <p/>
 * Refer to "A Practical Guide to Training Restricted Boltzmann Machines, Hinton 2010".
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

    private void sample_h_given_v(DoubleMatrix v_sample, DoubleMatrix h_probability, DoubleMatrix h_sample, DoubleMatrix curr_w, DoubleMatrix curr_hbias) {
        h_probability.copy(v_sample.mmul(curr_w.transpose()).addiRowVector(curr_hbias));
        MathUtil.sigmod(h_probability);
		if (null != h_sample) {
			for (int i = 0; i < h_probability.rows; i++) {
				for (int j = 0; j < h_probability.columns; j++) {
					h_sample.put(i, j, MathUtil.binomial(1, h_probability.get(i, j)));
				}
			}
		}
    }

    private void sample_v_given_h(DoubleMatrix h_sample, DoubleMatrix v_probability, DoubleMatrix v_sample, DoubleMatrix curr_w, DoubleMatrix curr_vbias) {
        v_probability.copy(h_sample.mmul(curr_w).addiRowVector(curr_vbias));
        MathUtil.sigmod(v_probability);
		if (null != v_sample) {
			for (int i = 0; i < v_probability.rows; i++) {
				for (int j = 0; j < v_probability.columns; j++) {
					v_sample.put(i, j, MathUtil.binomial(1, v_probability.get(i, j)));
				}
			}
		}
    }

    @Override
    protected void gradientUpdateMiniBatch(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param) {
    	int nbr_sample = x_samples.getRows();
    	DoubleMatrix curr_w = ((HiddenLayerParam)curr_param).w;
    	DoubleMatrix curr_hbias = ((HiddenLayerParam)curr_param).hbias;
    	DoubleMatrix curr_vbias = ((HiddenLayerParam)curr_param).vbias;
    	
    	DoubleMatrix v1_sample = x_samples;
    	DoubleMatrix h1_probability = new DoubleMatrix(nbr_sample, n_hidden);
    	DoubleMatrix h1_sample = new DoubleMatrix(nbr_sample, n_hidden);
    	DoubleMatrix v2_probability = new DoubleMatrix(nbr_sample, n_visible);
    	DoubleMatrix v2_sample = new DoubleMatrix(nbr_sample, n_visible);
    	DoubleMatrix h2_probability = new DoubleMatrix(nbr_sample, n_hidden);
    	//DoubleMatrix nh_samples = new DoubleMatrix(nbr_sample, n_hidden);
    	
    	sample_h_given_v(v1_sample, h1_probability, h1_sample, curr_w, curr_hbias);
    	if(config.isUseHintonCD1()) {
        	sample_v_given_h(h1_sample, v2_probability, null, curr_w, curr_vbias);
        	sample_h_given_v(v2_probability, h2_probability, null, curr_w, curr_hbias);
        }
        else {
            sample_v_given_h(h1_sample, v2_probability, v2_sample, curr_w, curr_vbias);
            sample_h_given_v(v2_sample, h2_probability, null, curr_w, curr_hbias);
        }
    	
    	DoubleMatrix delta_w = null;
    	DoubleMatrix delta_hbias = null;
    	DoubleMatrix delta_vbias = null;
    	
    	if(config.isUseHintonCD1()) {
    		delta_w = h1_probability.transpose().mmul(v1_sample).subi(h2_probability.transpose().mmul(v2_probability));
    		delta_hbias = h1_probability.sub(h2_probability).columnSums().divi(nbr_sample);
    		delta_vbias = v1_sample.sub(v2_probability).columnSums().divi(nbr_sample);
    	}
    	else {
    		delta_w = h1_sample.transpose().mmul(v1_sample).subi(h2_probability.transpose().mmul(v2_sample));
    		delta_hbias = h1_sample.sub(h2_probability).columnSums().divi(nbr_sample);
    		delta_vbias = v1_sample.sub(v2_sample).columnSums().divi(nbr_sample);
    	}

        if (config.isUseRegularization()) {
			//only L2 for RBM
			if (0 != config.getLamada2()) {
				delta_w.subi(curr_w.mul(config.getLamada2()));
            }
		}
        delta_w.divi(nbr_sample);
        
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
        
        //sample in hidden layer
        for (int i = 0; i < ret.rows; i++) {
			for (int j = 0; j < ret.columns; j++) {
				ret.put(i, j, MathUtil.binomial(1, ret.get(i, j)));
			}
		}
        
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
    	private DoubleMatrix v1_sample;
        private DoubleMatrix h1_probability;
        private DoubleMatrix h1_sample;
        private DoubleMatrix v2_probability;
        private DoubleMatrix v2_sample;
        private DoubleMatrix h2_probability;
        //private DoubleMatrix h2_sample;

        public RBMOptimizer(SGDTrainConfig config, DoubleMatrix samples, int n_visible, int n_hidden, DoubleMatrix w,
                DoubleMatrix hbias, DoubleMatrix vbias) {
            super(config, samples, n_visible, n_hidden, w, hbias, vbias);
            v1_sample = my_samples;
            h1_probability = new DoubleMatrix(nbr_sample, n_hidden);
            h1_sample = new DoubleMatrix(nbr_sample, n_hidden);
            v2_probability = new DoubleMatrix(nbr_sample, n_visible);
            v2_sample = new DoubleMatrix(nbr_sample, n_visible);
            h2_probability = new DoubleMatrix(nbr_sample, n_hidden);
            //h2_sample = new DoubleMatrix(nbr_sample, n_hidden);
        }

        @Override
        public double getValue() {
            sample_h_given_v(v1_sample, h1_probability, h1_sample, my_w, my_hbias);
            if(myConfig.isUseHintonCD1()) {
            	sample_v_given_h(h1_sample, v2_probability, null, my_w, my_vbias);
            }
            else {
                sample_v_given_h(h1_sample, v2_probability, v2_sample, my_w, my_vbias);
            }
            double loss = MatrixFunctions.powi(v1_sample.sub(v2_probability), 2).sum() / nbr_sample;
            if(myConfig.isUseRegularization()) {
            	//only use L2 for RBM
            	if(0 != myConfig.getLamada2()) {
            		loss += 0.5 * myConfig.getLamada2() * MatrixFunctions.pow(my_w, 2).sum();
            	}
            }
            return -loss;
        }

        @Override
        public void getValueGradient(double[] arg) {
        	if(myConfig.isUseHintonCD1()) {
                sample_h_given_v(v2_probability, h2_probability, null, my_w, my_hbias);
        	}
        	else {
        		sample_h_given_v(v2_sample, h2_probability, null, my_w, my_hbias);
        	}
        	
        	DoubleMatrix delta_w = null;
        	DoubleMatrix delta_hbias = null;
        	DoubleMatrix delta_vbias = null;
        	
        	if(myConfig.isUseHintonCD1()) {
        		delta_w = h1_probability.transpose().mmul(v1_sample).subi(h2_probability.transpose().mmul(v2_probability));
        		delta_hbias = h1_probability.sub(h2_probability).columnSums().divi(nbr_sample);
        		delta_vbias = v1_sample.sub(v2_probability).columnSums().divi(nbr_sample);
        	}
        	else {
        		delta_w = h1_sample.transpose().mmul(v1_sample).subi(h2_probability.transpose().mmul(v2_sample));
        		delta_hbias = h1_sample.sub(h2_probability).columnSums().divi(nbr_sample);
        		delta_vbias = v1_sample.sub(v2_sample).columnSums().divi(nbr_sample);
        	}

            if (myConfig.isUseRegularization()) {
				//only L2 for RBM
				if (0 != myConfig.getLamada2()) {
					delta_w.subi(my_w.mul(myConfig.getLamada2()));
                }
			}
            delta_w.divi(nbr_sample);
            
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
