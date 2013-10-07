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
package org.gd.spark.opendl.downpourSGD.bp;

import org.apache.log4j.Logger;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.train.SGDParam;
import org.gd.spark.opendl.util.MathUtil;
import org.gd.spark.opendl.util.MyConjugateGradient;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * AutoEncoder with tradition BP algorithm, without tied weights <p/>
 * refer to http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity
 * 
 * @author GuoDing
 * @since 2013-10-07
 */
public class AutoEncoder extends BP {
	private static final long serialVersionUID = 1L;
	private static final Logger logger = Logger.getLogger(AutoEncoder.class);
	
	public AutoEncoder(int _visible, int _hidden) {
		this(_visible, _hidden, null, null);
	}
	
	public AutoEncoder(int _visible, int _hidden, DoubleMatrix[] _w, DoubleMatrix[] _b) {
		super(_visible, _hidden, _w, _b);
	}
	
	/**
	 * Reconstruct, in fact same with super sigmod output
	 * @param input
	 * @return
	 */
	public final DoubleMatrix reconstruct(DoubleMatrix input) {
		return super.sigmod_output(input);
	}
	
	/**
	 * Reconstruct, in fact same with super sigmod output
	 * @param input
	 * @return
	 */
	public final void reconstruct(double[] input, double[] output) {
		super.sigmod_output(input, output);
	}

	@Override
	protected boolean isSupervise() {
		return false;
	}
	
	@Override
	protected void gradientUpdateMiniBatch(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param) {
		int nbr_sample = x_samples.rows;
		BPParam curr_pbparam = (BPParam)curr_param;
		DoubleMatrix[] activation = new DoubleMatrix[curr_pbparam.nl];
		DoubleMatrix[] l_bias = new DoubleMatrix[curr_pbparam.nl];
		DoubleMatrix avg_hidden = null;
		
		/**
		 * feedforward
		 */
		activation[0] = x_samples;
		for(int i = 1; i < curr_pbparam.nl; i++) {
			activation[i] = activation[i - 1].mmul(curr_pbparam.w[i - 1].transpose()).addiRowVector(curr_pbparam.b[i - 1]);
			MathUtil.sigmod(activation[i]);
		}
		//sparsity
		if(config.isForceSparsity()) {
			avg_hidden = activation[1].columnSums().divi(nbr_sample);
		}
		
		/**
		 * backward
		 */
		// 1 last layer
		DoubleMatrix ai = activation[curr_pbparam.nl - 1];
		l_bias[curr_pbparam.nl - 1] = ai.sub(x_samples).muli(ai).muli(ai.neg().addi(1));
		
		//2 back
		for(int i = curr_pbparam.nl - 2; i >= 1; i--) {
			l_bias[i] = l_bias[i + 1].mmul(curr_pbparam.w[i]);
			if(config.isForceSparsity()) {
				DoubleMatrix sparsity_v = avg_hidden.dup();
				for(int k = 0; k < sparsity_v.columns; k++) {
					double roat = config.getSparsity();
					double roat_k = sparsity_v.get(0, k);
					sparsity_v.put(0, k, config.getSparsityBeta()*((1-roat)/(1-roat_k) - roat/roat_k));
				}
				l_bias[i].addiRowVector(sparsity_v);
			}
			ai = activation[i];
			l_bias[i].muli(ai).muli(ai.neg().addi(1));
		}
		
		/**
		 * delta
		 */
		for(int i = 0; i < curr_pbparam.w.length; i++) {
			DoubleMatrix delta_wi = l_bias[i + 1].transpose().mmul(activation[i]).divi(nbr_sample);
			if(config.isUseRegularization()) {
				//for bp, only use L2
				if(0 != config.getLamada2()) {
				    delta_wi.addi(curr_pbparam.w[i].mul(config.getLamada2()));
				}
			}
			curr_pbparam.w[i].subi(delta_wi.muli(config.getLearningRate()));
		}
		for(int i = 0; i < curr_pbparam.b.length; i++) {
			DoubleMatrix delta_bi = l_bias[i + 1].columnSums().divi(nbr_sample);
			curr_pbparam.b[i].subi(delta_bi.transpose().muli(config.getLearningRate()));
		}
	}
	
	@Override
	protected void gradientUpdateCG(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param) {
		AEOptimizer opt = new AEOptimizer(config, x_samples, (BPParam)curr_param);
        MyConjugateGradient cg = new MyConjugateGradient(opt, config.getCgInitStepSize());
        cg.setTolerance(config.getCgTolerance());
        try {
            cg.optimize(config.getCgMaxIterations());
        } catch (Throwable e) {
            logger.error("", e);
        } 
	}
	
	private class AEOptimizer extends BP.BPOptimizer {
		private DoubleMatrix avg_hidden;

		public AEOptimizer(SGDTrainConfig config, DoubleMatrix x_samples, BPParam curr_bpparam) {
			super(config, x_samples, null, curr_bpparam);
		}
		
		@Override
		public double getValue() {
			/**
			 * feedforward
			 */
			for(int i = 1; i < my_bpparam.nl; i++) {
				activation[i] = activation[i - 1].mmul(my_bpparam.w[i - 1].transpose()).addiRowVector(my_bpparam.b[i - 1]);
				MathUtil.sigmod(activation[i]);
			}
			double loss = MatrixFunctions.powi(activation[my_bpparam.nl - 1].sub(my_x_samples), 2).sum() / nbr_samples;
			
			//regulation
			if (my_config.isUseRegularization()) {
				//only L2 for BP
				if (0 != my_config.getLamada2()) {
                    double sum_square_w = 0;
                    for(int i = 0; i < my_bpparam.w.length; i++) {
                    	sum_square_w += MatrixFunctions.pow(my_bpparam.w[i], 2).sum();
                    }
                    loss += 0.5 * my_config.getLamada2() * sum_square_w;
                }
			}
			
			//sparsity
			if(my_config.isForceSparsity()) {
				avg_hidden = activation[1].columnSums().divi(nbr_samples);
				double kl = 0;
				for(int i = 0; i < n_hiddens[0]; i++) {
					kl += my_config.getSparsity() * Math.log(my_config.getSparsity()/avg_hidden.get(0, i));
					kl += (1 - my_config.getSparsity()) * Math.log((1 - my_config.getSparsity())/(1-avg_hidden.get(0, i)));
				}
				loss += my_config.getSparsityBeta() * kl;
			}
			
			return -loss;
		}

		@Override
		public void getValueGradient(double[] arg) {
			DoubleMatrix[] l_bias = new DoubleMatrix[my_bpparam.nl];

			/**
			 * backward
			 */
			// 1 last layer
			DoubleMatrix ai = activation[my_bpparam.nl - 1];
			l_bias[my_bpparam.nl - 1] = ai.sub(my_y_samples).muli(ai).muli(ai.neg().addi(1));
			
			//2 back(no layer0 error need)
			for(int i = my_bpparam.nl - 2; i >= 1; i--) {
				l_bias[i] = l_bias[i + 1].mmul(my_bpparam.w[i]);
				if(my_config.isForceSparsity()) {
					DoubleMatrix sparsity_v = avg_hidden.dup();
					for(int k = 0; k < sparsity_v.columns; k++) {
						double roat = my_config.getSparsity();
						double roat_k = sparsity_v.get(0, k);
						sparsity_v.put(0, k, my_config.getSparsityBeta()*((1-roat)/(1-roat_k) - roat/roat_k));
					}
					l_bias[i].addiRowVector(sparsity_v);
				}
				ai = activation[i];
				l_bias[i].muli(ai).muli(ai.neg().addi(1));
			}
			
			/**
			 * delta
			 */
			int idx = 0;
			for(int i = 0; i < my_bpparam.w.length; i++) {
				DoubleMatrix delta_wi = l_bias[i + 1].transpose().mmul(activation[i]).divi(nbr_samples);
				if(my_config.isUseRegularization()) {
					//for bp, only use L2
					if(0 != my_config.getLamada2()) {
					    delta_wi.addi(my_bpparam.w[i].mul(my_config.getLamada2()));
					}
				}
				for(int row = 0; row < delta_wi.rows; row++) {
					for(int col = 0; col < delta_wi.columns; col++) {
						arg[idx++] = -delta_wi.get(row, col);
					}
				}
			}
			for(int i = 0; i < my_bpparam.b.length; i++) {
				DoubleMatrix delta_bi = l_bias[i + 1].columnSums().divi(nbr_samples);
				for(int row = 0; row < delta_bi.rows; row++) {
					for(int col = 0; col < delta_bi.columns; col++) {
						arg[idx++] = -delta_bi.get(row, col);
					}
				}
			}
		}
	}
}
