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
package org.gd.spark.opendl.downpourSGD.Backpropagation;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Writer;
import java.util.List;

import org.apache.log4j.Logger;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.downpourSGD.train.SGDBase;
import org.gd.spark.opendl.downpourSGD.train.SGDParam;
import org.gd.spark.opendl.util.MathUtil;
import org.gd.spark.opendl.util.MyConjugateGradient;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import cc.mallet.optimize.Optimizable;

/**
 * Backpropagation Algorithm <p/>
 * refer to http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
 * 
 * @author GuoDing
 * @since 2013-10-06
 */
public class BP extends SGDBase {
	private static final long serialVersionUID = 1L;
	private static final Logger logger = Logger.getLogger(BP.class);
	
	protected int n_in;
	protected int n_out;
	protected int[] n_hiddens;
	protected BPParam bpparam;
	
	public BP(int _in, int _out, int[] _hiddens) {
		this(_in, _out, _hiddens, null, null);
	}
	
	/**
	 * Construct for auto encoder subclass
	 * @param _visible
	 * @param int_hidden
	 */
	protected BP(int _visible, int _hidden) {
		this(_visible, _hidden, null, null);
	}
	
	/**
	 * Construct for auto encoder subclass with initial weights and bias
	 * @param _visible
	 * @param int_hidden
	 */
	protected BP(int _visible, int _hidden, DoubleMatrix[] _w, DoubleMatrix[] _b) {
		n_in = _visible;
		n_out = _visible;
		n_hiddens = new int[1];
		n_hiddens[0] = _hidden;
		bpparam = new BPParam(n_in, n_out, n_hiddens, _w, _b);
		param = bpparam;
	}
	
	public BP(int _in, int _out, int[] _hiddens, DoubleMatrix[] _w, DoubleMatrix[] _b) {
		n_in = _in;
		n_out = _out;
		n_hiddens = new int[_hiddens.length];
		for(int i = 0; i < _hiddens.length; i++) {
			n_hiddens[i] = _hiddens[i];
		}
		bpparam = new BPParam(_in, _out, _hiddens, _w, _b);
		param = bpparam;
	}
	
	/**
     * Sigmod output(standalone)
     * @param input Input layer matrix
     * @return Output layer output matrix
     */
	public final DoubleMatrix sigmod_output(DoubleMatrix input) {
		DoubleMatrix output = input;
		for(int i = 0; i < bpparam.w.length; i++) {
			output = output.mmul(bpparam.w[i].transpose()).addiRowVector(bpparam.b[i]);
			MathUtil.sigmod(output);
		}
		return output;
	}
	
	/**
	 * Sigmod output(standalone)
	 * @param input Input layer data
	 * @param output Output layer data
	 */
	public final void sigmod_output(double[] input, double[] output) {
		DoubleMatrix input_m = new DoubleMatrix(input).transpose();
		DoubleMatrix output_m = sigmod_output(input_m);
		for(int i = 0; i < output.length; i++) {
			output[i] = output_m.get(0, i);
		}
	}

	@Override
	public void read(DataInput in) throws IOException {
		n_in = in.readInt();
		n_out = in.readInt();
		for(int i = 0; i < n_hiddens.length; i++) {
			n_hiddens[i] = in.readInt();
		}
		for(int i = 0; i < n_hiddens.length; i++) {
			for(int j = 0; j < bpparam.w[i].rows; j++) {
				for(int k = 0; k < bpparam.w[i].columns; k++) {
					bpparam.w[i].put(j, k, in.readDouble());
				}
			}
			for(int j = 0; j < bpparam.b[i].rows; j++) {
				for(int k = 0; k < bpparam.b[i].columns; k++) {
					bpparam.b[i].put(j, k, in.readDouble());
				}
			}
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(n_in);
		out.writeInt(n_out);
		for(int i = 0; i < n_hiddens.length; i++) {
			out.writeInt(n_hiddens[i]);
		}
		for(int i = 0; i < n_hiddens.length; i++) {
			for(int j = 0; j < bpparam.w[i].rows; j++) {
				for(int k = 0; k < bpparam.w[i].columns; k++) {
					out.writeDouble(bpparam.w[i].get(j, k));
				}
			}
			for(int j = 0; j < bpparam.b[i].rows; j++) {
				for(int k = 0; k < bpparam.b[i].columns; k++) {
					out.writeDouble(bpparam.b[i].get(j, k));
				}
			}
		}
	}

	@Override
	public void print(Writer wr) throws IOException {
		String newLine = System.getProperty("line.separator");
		wr.write(String.valueOf(n_in));
        wr.write(",");
        wr.write(String.valueOf(n_out));
        wr.write(newLine);
        for(int i = 0; i < n_hiddens.length; i++) {
			wr.write(String.valueOf(n_hiddens[i]));
			wr.write(",");
		}
        wr.write(newLine);
        for(int i = 0; i < n_hiddens.length; i++) {
			for(int j = 0; j < bpparam.w[i].rows; j++) {
				for(int k = 0; k < bpparam.w[i].columns; k++) {
					wr.write(String.valueOf(bpparam.w[i].get(j, k)));
					wr.write(",");
				}
				wr.write(newLine);
			}
			wr.write(newLine);
			for(int j = 0; j < bpparam.b[i].rows; j++) {
				for(int k = 0; k < bpparam.b[i].columns; k++) {
					wr.write(String.valueOf(bpparam.b[i].get(j, k)));
					wr.write(",");
				}
				wr.write(newLine);
			}
			wr.write(newLine);
		}
        wr.write(newLine);
	}

	@Override
	protected void gradientUpdateMiniBatch(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param) {
		int nbr_sample = x_samples.rows;
		BPParam curr_pbparam = (BPParam)curr_param;
		DoubleMatrix[] activation = new DoubleMatrix[curr_pbparam.nl];
		DoubleMatrix[] l_bias = new DoubleMatrix[curr_pbparam.nl];
		
		/**
		 * feedforward
		 */
		activation[0] = x_samples;
		for(int i = 1; i < curr_pbparam.nl; i++) {
			activation[i] = activation[i - 1].mmul(curr_pbparam.w[i - 1].transpose()).addiRowVector(curr_pbparam.b[i - 1]);
			MathUtil.sigmod(activation[i]);
		}
		
		/**
		 * backward
		 */
		// 1 last layer
		DoubleMatrix ai = activation[curr_pbparam.nl - 1];
		l_bias[curr_pbparam.nl - 1] = ai.sub(y_samples).muli(ai).muli(ai.neg().addi(1));
		
		//2 back
		for(int i = curr_pbparam.nl - 2; i >= 1; i--) {
			ai = activation[i];
			l_bias[i] = l_bias[i + 1].mmul(curr_pbparam.w[i]).muli(ai).muli(ai.neg().addi(1));
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
		BPOptimizer lropt = new BPOptimizer(config, x_samples, y_samples, (BPParam)curr_param);
        MyConjugateGradient cg = new MyConjugateGradient(lropt, config.getCgInitStepSize());
        cg.setTolerance(config.getCgTolerance());
        try {
            cg.optimize(config.getCgMaxIterations());
        } catch (Throwable e) {
            logger.error("", e);
        } 
	}

	@Override
	protected void mergeParam(SGDParam new_param, int nrModelReplica) {
		BPParam new_bpparam = (BPParam)new_param;
		for(int i = 0; i < bpparam.w.length; i++) {
			bpparam.w[i].addi(new_bpparam.w[i].sub(bpparam.w[i]).divi(nrModelReplica));
		}
		for(int i = 0; i < bpparam.b.length; i++) {
			bpparam.b[i].addi(new_bpparam.b[i].sub(bpparam.b[i]).divi(nrModelReplica));
		}
	}

	@Override
	protected double loss(List<SampleVector> samples) {
		DoubleMatrix x_samples = MathUtil.convertX2Matrix(samples);
        DoubleMatrix y_samples = MathUtil.convertY2Matrix(samples);
        DoubleMatrix sigmod_output = sigmod_output(x_samples);
		return MatrixFunctions.powi(sigmod_output.sub(y_samples), 2).sum();
	}

	@Override
	protected boolean isSupervise() {
		return true;
	}

	protected static class BPOptimizer implements Optimizable.ByGradientValue {
		protected int nbr_samples;
		protected SGDTrainConfig my_config;
		protected DoubleMatrix my_x_samples;
		protected DoubleMatrix my_y_samples;
		protected BPParam my_bpparam;
		protected DoubleMatrix[] activation;
        
        public BPOptimizer(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, BPParam curr_bpparam) {
        	my_x_samples = x_samples;
            my_y_samples = y_samples;
            my_bpparam = curr_bpparam;
            nbr_samples = x_samples.getRows();
            my_config = config;
            activation = new DoubleMatrix[my_bpparam.nl];
            activation[0] = my_x_samples;
        }
        
		@Override
		public int getNumParameters() {
			int ret = 0;
			for(int i = 0; i < my_bpparam.w.length; i++) {
				ret += my_bpparam.w[i].length;
			}
			for(int i = 0; i < my_bpparam.b.length; i++) {
				ret += my_bpparam.b[i].length;
			}
			return ret;
		}

		@Override
		public double getParameter(int arg) {
			int total = 0;
			for(int i = 0; i < my_bpparam.w.length; i++) {
				if(arg < (total + my_bpparam.w[i].length)) {
					int row = (arg - total) / my_bpparam.w[i].columns;
					int col = (arg - total) % my_bpparam.w[i].columns;
					return my_bpparam.w[i].get(row, col);
				}
				else {
					total += my_bpparam.w[i].length;
					continue;
				}
			}
			for(int i = 0; i < my_bpparam.b.length; i++) {
				if(arg < (total + my_bpparam.b[i].length)) {
					int row = (arg - total) / my_bpparam.b[i].columns;
					int col = (arg - total) % my_bpparam.b[i].columns;
					return my_bpparam.b[i].get(row, col);
				}
				else {
					total += my_bpparam.b[i].length;
					continue;
				}
			}
			return 0;
		}

		@Override
		public void getParameters(double[] arg) {
			int idx = 0;
			for(int i = 0; i < my_bpparam.w.length; i++) {
				for(int row = 0; row < my_bpparam.w[i].rows; row++) {
					for(int col = 0; col < my_bpparam.w[i].columns; col++) {
						arg[idx++] = my_bpparam.w[i].get(row, col);
					}
				}
			}
			for(int i = 0; i < my_bpparam.b.length; i++) {
				for(int row = 0; row < my_bpparam.b[i].rows; row++) {
					for(int col = 0; col < my_bpparam.b[i].columns; col++) {
						arg[idx++] = my_bpparam.b[i].get(row, col);
					}
				}
			}
		}

		@Override
		public void setParameter(int arg0, double arg1) {
			int total = 0;
			for(int i = 0; i < my_bpparam.w.length; i++) {
				if(arg0 < (total + my_bpparam.w[i].length)) {
					int row = (arg0 - total) / my_bpparam.w[i].columns;
					int col = (arg0 - total) % my_bpparam.w[i].columns;
					my_bpparam.w[i].put(row, col, arg1);
					return;
				}
				else {
					total += my_bpparam.w[i].length;
					continue;
				}
			}
			for(int i = 0; i < my_bpparam.b.length; i++) {
				if(arg0 < (total + my_bpparam.b[i].length)) {
					int row = (arg0 - total) / my_bpparam.b[i].columns;
					int col = (arg0 - total) % my_bpparam.b[i].columns;
					my_bpparam.b[i].put(row, col, arg1);
					return;
				}
				else {
					total += my_bpparam.b[i].length;
					continue;
				}
			}
		}

		@Override
		public void setParameters(double[] arg) {
			int idx = 0;
			for(int i = 0; i < my_bpparam.w.length; i++) {
				for(int row = 0; row < my_bpparam.w[i].rows; row++) {
					for(int col = 0; col < my_bpparam.w[i].columns; col++) {
						my_bpparam.w[i].put(row, col, arg[idx++]);
					}
				}
			}
			for(int i = 0; i < my_bpparam.b.length; i++) {
				for(int row = 0; row < my_bpparam.b[i].rows; row++) {
					for(int col = 0; col < my_bpparam.b[i].columns; col++) {
						my_bpparam.b[i].put(row, col, arg[idx++]);
					}
				}
			}
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
			double loss = MatrixFunctions.powi(activation[my_bpparam.nl - 1].sub(my_y_samples), 2).sum() / nbr_samples;
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
				ai = activation[i];
				l_bias[i] = l_bias[i + 1].mmul(my_bpparam.w[i]).muli(ai).muli(ai.neg().addi(1));
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
