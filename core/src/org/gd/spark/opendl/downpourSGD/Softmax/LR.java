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
package org.gd.spark.opendl.downpourSGD.Softmax;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Writer;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
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
 * Logistic Regression(Softmax) node <p/>
 * 
 * @author GuoDing
 * @since 2013-08-01
 */
public final class LR extends SGDBase {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(LR.class);
    private int x_num;
    private int y_num;
    private LRParam lrparam;

    /**
     * Constructor with random initial W&&B param
     * 
     * @param x_feature_num Feature num
     * @param y_class_num Class num
     */
    public LR(int x_feature_num, int y_class_num) {
        this(x_feature_num, y_class_num, null, null);
    }

    /**
     * Constructor with initial W&&B param
     * 
     * @param x_feature_num Feature num
     * @param y_class_num Class num
     * @param _w Specify W param matrix
     * @param _b Specify B param vector
     */
    public LR(int x_feature_num, int y_class_num, double[][] _w, double[] _b) {
        x_num = x_feature_num;
        y_num = y_class_num;
        lrparam = new LRParam(x_feature_num, y_class_num, _w, _b);
        param = lrparam;
    }

    /**
     * Do predict work with multiple sample(standalone)
     * @param x Input samples matrix
     * @return Predict result matrix (row=x's row, column=class num)
     */
    public final DoubleMatrix predict(DoubleMatrix x) {
        DoubleMatrix y = x.mmul(lrparam.w.transpose()).addiRowVector(lrparam.b);
        softmax(y);
        return y;
    }

    /**
     * Do predict work with one sample
     * @param x Input sample
     * @param y Output predict result
     */
    public final void predict(double[] x, double[] y) {
        for (int i = 0; i < y_num; i++) {
            y[i] = 0;
            for (int j = 0; j < x_num; j++) {
                y[i] += lrparam.w.get(i, j) * x[j];
            }
            y[i] += lrparam.b.get(i, 0);
        }
        softmax(y);
    }
    
    /**
     * Do predict work on spark
     * @param samples Input data RDD
     * @copyX Whether copy x data from original input to output SampleVector
     * @return Predict result data RDD
     */
    public final JavaRDD<SampleVector> predict(JavaRDD<SampleVector> samples, boolean copyX) {
    	return samples.map(new PredictSpark(copyX));
    }

    private void softmax(DoubleMatrix y) {
        DoubleMatrix max = y.rowMaxs();
        MatrixFunctions.expi(y.subiColumnVector(max));
        DoubleMatrix sum = y.rowSums();
        y.diviColumnVector(sum);
    }

    private void softmax(double[] y) {
        double max = 0.0;
        double sum = 0.0;
        for (int i = 0; i < y_num; i++) {
            if (max < y[i]) {
                max = y[i];
            }
        }
        for (int i = 0; i < y_num; i++) {
            y[i] = Math.exp(y[i] - max);
            sum += y[i];
        }
        for (int i = 0; i < y_num; i++) {
            y[i] /= sum;
        }
    }

    /**
     * 
     * @return W param matrix
     */
    public DoubleMatrix getW() {
        return lrparam.w;
    }

    /**
     * 
     * @return B param vector
     */
    public DoubleMatrix getB() {
        return lrparam.b;
    }

    /**
     * 
     * @return Feature num
     */
    public int getX() {
        return x_num;
    }

    /**
     * 
     * @return Class num
     */
    public int getY() {
        return y_num;
    }

    @Override
    public final void read(DataInput in) throws IOException {
        x_num = in.readInt();
        y_num = in.readInt();
        for (int i = 0; i < y_num; i++) {
            for (int j = 0; j < x_num; j++) {
            	lrparam.w.put(i, j, in.readDouble());
            }
        }
        for (int i = 0; i < y_num; i++) {
        	lrparam.b.put(i, 0, in.readDouble());
        }
    }

    @Override
    public final void write(DataOutput out) throws IOException {
        out.writeInt(x_num);
        out.writeInt(y_num);
        for (int i = 0; i < y_num; i++) {
            for (int j = 0; j < x_num; j++) {
                out.writeDouble(lrparam.w.get(i, i));
            }
        }
        for (int i = 0; i < y_num; i++) {
            out.writeDouble(lrparam.b.get(i, 0));
        }
    }

    @Override
    public final void print(Writer wr) throws IOException {
        String newLine = System.getProperty("line.separator");
        wr.write(String.valueOf(x_num));
        wr.write(",");
        wr.write(String.valueOf(y_num));
        wr.write(newLine);
        for (int i = 0; i < y_num; i++) {
            for (int j = 0; j < x_num; j++) {
                wr.write(String.valueOf(lrparam.w.get(i, i)));
                wr.write(",");
            }
            wr.write(newLine);
        }
        for (int i = 0; i < y_num; i++) {
            wr.write(String.valueOf(lrparam.b.get(i, 0)));
            wr.write(",");
        }
        wr.write(newLine);
    }

    private class LROptimizer implements Optimizable.ByGradientValue {
        private DoubleMatrix my_w;
        private DoubleMatrix my_b;
        private DoubleMatrix my_x_samples;
        private DoubleMatrix my_y_samples;
        private DoubleMatrix curr_predict_y;
        private int nbr_samples;
        private SGDTrainConfig my_config;

        public LROptimizer(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, DoubleMatrix curr_w,
                DoubleMatrix curr_b) {
            my_x_samples = x_samples;
            my_y_samples = y_samples;
            my_w = curr_w;
            my_b = curr_b;
            nbr_samples = x_samples.getRows();
            my_config = config;
        }

        @Override
        public int getNumParameters() {
            return y_num * x_num + y_num;
        }

        @Override
        public double getParameter(int arg) {
            if (arg < y_num * x_num) {
                int i = arg / x_num;
                int j = arg % x_num;
                return my_w.get(i, j);
            }
            return my_b.get(arg - y_num * x_num, 0);
        }

        @Override
        public void getParameters(double[] arg) {
            int idx = 0;
            for (int i = 0; i < y_num; i++) {
                for (int j = 0; j < x_num; j++) {
                    arg[idx++] = my_w.get(i, j);
                }
            }
            for (int i = 0; i < y_num; i++) {
                arg[idx++] = my_b.get(i, 0);
            }
        }

        @Override
        public void setParameter(int arg0, double arg1) {
            if (arg0 < y_num * x_num) {
                int i = arg0 / x_num;
                int j = arg0 % x_num;
                my_w.put(i, j, arg1);
            } else {
                my_b.put(arg0 - y_num * x_num, 0, arg1);
            }
        }

        @Override
        public void setParameters(double[] arg) {
            int idx = 0;
            for (int i = 0; i < y_num; i++) {
                for (int j = 0; j < x_num; j++) {
                    my_w.put(i, j, arg[idx++]);
                }
            }
            for (int i = 0; i < y_num; i++) {
                my_b.put(i, 0, arg[idx++]);
            }
        }

        @Override
        public double getValue() {
            curr_predict_y = my_x_samples.mmul(my_w.transpose()).addiRowVector(my_b);
            softmax(curr_predict_y);
            double loss = MatrixFunctions.powi(curr_predict_y.sub(my_y_samples), 2).sum() / nbr_samples;
            if (my_config.isUseRegularization()) {
                if (0 != my_config.getLamada1()) {
                    loss += my_config.getLamada1()
                            * (MatrixFunctions.abs(my_w).sum() + MatrixFunctions.abs(my_b).sum()); // L1
                }
                if (0 != my_config.getLamada2()) {
                    loss += 0.5 * my_config.getLamada2()
                            * (MatrixFunctions.pow(my_w, 2).sum() + MatrixFunctions.pow(my_b, 2).sum()); // L2
                }
            }
            return -loss;
        }

        @Override
        public void getValueGradient(double[] arg) {
            DoubleMatrix delta_b = my_y_samples.sub(curr_predict_y);
            DoubleMatrix delta_w = delta_b.transpose().mmul(my_x_samples);
            delta_b = delta_b.columnSums().divi(nbr_samples);
            delta_w.divi(nbr_samples);

            if (my_config.isUseRegularization()) {
                if (0 != my_config.getLamada1()) {
                    delta_w.addi(MatrixFunctions.signum(my_w).mmuli(my_config.getLamada1()));
                    delta_b.addi(MatrixFunctions.signum(my_b).transpose().mmuli(my_config.getLamada1()));
                }
                if (0 != my_config.getLamada2()) {
                    delta_w.addi(my_w.mmul(my_config.getLamada2()));
                    delta_b.addi(my_b.transpose().mmul(my_config.getLamada2()));
                }
            }

            int idx = 0;
            for (int i = 0; i < y_num; i++) {
                for (int j = 0; j < x_num; j++) {
                    arg[idx++] = delta_w.get(i, j);
                }
            }
            for (int i = 0; i < y_num; i++) {
                arg[idx++] = delta_b.get(0, i);
            }
        }
    }
    
    private class PredictSpark extends Function<SampleVector, SampleVector> {
		private static final long serialVersionUID = 1L;
		private boolean copyX = false;
		public PredictSpark(boolean _copyX) {
			copyX = _copyX;
		}
		@Override
		public SampleVector call(SampleVector arg) throws Exception {
			SampleVector ret = new SampleVector(x_num, y_num);
			if(copyX) {
				for(int i = 0; i < x_num; i++) {
					ret.getX()[i] = arg.getX()[i];
				}
			}
			predict(arg.getX(), ret.getY());
			return ret;
		}
    }

	@Override
	protected void gradientUpdateMiniBatch(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param) {
		int nbr_samples = x_samples.rows;
		DoubleMatrix curr_w = ((LRParam)curr_param).w;
		DoubleMatrix curr_b = ((LRParam)curr_param).b;
		
    	DoubleMatrix curr_predict_y = x_samples.mmul(curr_w.transpose()).addiRowVector(curr_b);
        softmax(curr_predict_y);
        DoubleMatrix delta_b = y_samples.sub(curr_predict_y);
        DoubleMatrix delta_w = delta_b.transpose().mmul(x_samples);
        delta_b = delta_b.columnSums().divi(nbr_samples);
        delta_w.divi(nbr_samples);
        
        if (config.isUseRegularization()) {
            if (0 != config.getLamada1()) {
                delta_w.addi(MatrixFunctions.signum(curr_w).mmuli(config.getLamada1()));
                delta_b.addi(MatrixFunctions.signum(curr_b).transpose().mmuli(config.getLamada1()));
            }
            if (0 != config.getLamada2()) {
                delta_w.addi(curr_w.mmul(config.getLamada2()));
                delta_b.addi(curr_b.transpose().mmul(config.getLamada2()));
            }
        }
        
        curr_w.addi(delta_w.muli(config.getLearningRate()));
        curr_b.addi(delta_b.transpose().muli(config.getLearningRate()));
	}

	@Override
	protected void gradientUpdateCG(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param) {
		DoubleMatrix curr_w = ((LRParam)curr_param).w;
		DoubleMatrix curr_b = ((LRParam)curr_param).b;
		
		LROptimizer lropt = new LROptimizer(config, x_samples, y_samples, curr_w, curr_b);
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
		LRParam new_lrparam = (LRParam)new_param;
		lrparam.w.addi(new_lrparam.w.sub(lrparam.w).divi(nrModelReplica));
    	lrparam.b.addi(new_lrparam.b.sub(lrparam.b).divi(nrModelReplica));
	}

	@Override
	protected double loss(List<SampleVector> samples) {
		DoubleMatrix x_samples = MathUtil.convertX2Matrix(samples);
        DoubleMatrix y_samples = MathUtil.convertY2Matrix(samples);
        DoubleMatrix predict_y = predict(x_samples);
        return MatrixFunctions.powi(predict_y.sub(y_samples), 2).sum();
	}

	@Override
	protected boolean isSupervise() {
		return true;
	}
}
