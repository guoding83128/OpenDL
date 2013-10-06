package org.gd.spark.opendl.downpourSGD.old.hLayer;

import org.apache.log4j.Logger;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.util.MathUtil;
import org.gd.spark.opendl.util.MyConjugateGradient;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Denoising Autoencoders implementation <p/>
 * refer to http://deeplearning.net/tutorial/dA.html
 * 
 * @author GuoDing
 * @since 2013-08-15
 */
public class dA extends HiddenLayer {
    private static final Logger logger = Logger.getLogger(dA.class);
    private static final long serialVersionUID = 1L;

    public dA(int in_n_visible, int in_n_hidden) {
        super(in_n_visible, in_n_hidden);
    }

    public dA(int in_n_visible, int in_n_hidden, double[][] in_w, double[] in_hbias, double[] in_vbias) {
        super(in_n_visible, in_n_hidden, in_w, in_hbias);
        if (null == in_vbias) {
            vbias = new DoubleMatrix(n_visible);
        } else {
            vbias = new DoubleMatrix(in_vbias);
        }
    }

    @Override
    protected void gradientUpdateMiniBatch(SGDTrainConfig config, DoubleMatrix samples, DoubleMatrix curr_w,
            DoubleMatrix curr_hbias, DoubleMatrix curr_vbias) {
    	int nbr_sample = samples.rows;
        
        /**
         * reconstruct
         */
    	DoubleMatrix tilde_x = null;
        DoubleMatrix y = null;
        DoubleMatrix z = null;
    	if (config.isDoCorruption()) {
            double p = 1 - config.getCorruption_level();
            tilde_x = get_corrupted_input(samples, p);
            y = tilde_x.mmul(curr_w.transpose()).addiRowVector(curr_hbias);
        }
    	else {
    		y = samples.mmul(curr_w.transpose()).addiRowVector(curr_hbias);
    	}
    	MathUtil.sigmod(y);
        z = y.mmul(curr_w).addiRowVector(curr_vbias);
        MathUtil.sigmod(z);
        
        /**
         * gradient update
         */
        DoubleMatrix L_vbias = samples.sub(z);
        DoubleMatrix L_hbias = L_vbias.mmul(curr_w.transpose()).muli(y).muli(y.neg().addi(1));
        DoubleMatrix delta_w = null;
        if (config.isDoCorruption()) {
            delta_w = L_hbias.transpose().mmul(tilde_x).addi(y.transpose().mmul(L_vbias));
        } else {
            delta_w = L_hbias.transpose().mmul(samples).addi(y.transpose().mmul(L_vbias));
        }
        
        delta_w.divi(nbr_sample);
        DoubleMatrix delta_hbias = L_hbias.columnSums().divi(nbr_sample);
        DoubleMatrix delta_vbias = L_vbias.columnSums().divi(nbr_sample);
        
        curr_w.addi(delta_w.muli(config.getLearningRate()));
        curr_hbias.addi(delta_hbias.transpose().muli(config.getLearningRate()));
        curr_vbias.addi(delta_vbias.transpose().muli(config.getLearningRate()));
    }

    @Override
    protected void gradientUpdateCG(SGDTrainConfig config, DoubleMatrix samples, DoubleMatrix curr_w,
            DoubleMatrix curr_hbias, DoubleMatrix curr_vbias) {
        dAOptimizer daopt = new dAOptimizer(config, samples, n_visible, n_hidden, curr_w, curr_hbias, curr_vbias);
        MyConjugateGradient cg = new MyConjugateGradient(daopt, config.getCgInitStepSize());
        cg.setTolerance(config.getCgTolerance());
        try {
            cg.optimize(config.getCgMaxIterations());
        } catch (Throwable e) {
            logger.error("", e);
        }
    }

    @Override
    public DoubleMatrix reconstruct(DoubleMatrix input) {
        DoubleMatrix ret = input.mmul(w.transpose()).addiRowVector(hbias);
        MathUtil.sigmod(ret);
        ret = ret.mmul(w).addiRowVector(vbias);
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

    private DoubleMatrix get_corrupted_input(DoubleMatrix x, double p) {
        DoubleMatrix ret = new DoubleMatrix(x.getRows(), x.getColumns());
        for (int i = 0; i < x.getRows(); i++) {
            for (int j = 0; j < x.getColumns(); j++) {
                if (0 != x.get(i, j)) {
                    ret.put(i, j, MathUtil.binomial(1, p));
                }
            }
        }
        return ret;
    }

    private class dAOptimizer extends HiddenLayerOptimizer {
        private DoubleMatrix tilde_x;
        private DoubleMatrix y;
        private DoubleMatrix z;

        public dAOptimizer(SGDTrainConfig config, DoubleMatrix samples, int n_visible, int n_hidden, DoubleMatrix w,
                DoubleMatrix hbias, DoubleMatrix vbias) {
            super(config, samples, n_visible, n_hidden, w, hbias, vbias);
            if (myConfig.isDoCorruption()) {
                double p = 1 - this.myConfig.getCorruption_level();
                tilde_x = get_corrupted_input(my_samples, p);
            }
        }

        @Override
        public double getValue() {
            if (myConfig.isDoCorruption()) {
                y = tilde_x.mmul(my_w.transpose()).addiRowVector(my_hbias);
            } else {
                y = my_samples.mmul(my_w.transpose()).addiRowVector(my_hbias);
            }
            MathUtil.sigmod(y);
            z = y.mmul(my_w).addiRowVector(my_vbias);
            MathUtil.sigmod(z);

            double loss = MatrixFunctions.powi(my_samples.sub(z), 2).sum() / nbr_sample;
            return -loss;
        }

        @Override
        public void getValueGradient(double[] arg) {
            DoubleMatrix L_vbias = my_samples.sub(z);
            DoubleMatrix L_hbias = L_vbias.mmul(my_w.transpose()).muli(y).muli(y.neg().addi(1));
            DoubleMatrix delta_w = null;
            if (myConfig.isDoCorruption()) {
                delta_w = L_hbias.transpose().mmul(tilde_x).addi(y.transpose().mmul(L_vbias));
            } else {
                delta_w = L_hbias.transpose().mmul(my_samples).addi(y.transpose().mmul(L_vbias));
            }
            delta_w.divi(nbr_sample);
            DoubleMatrix delta_hbias = L_hbias.columnSums().divi(nbr_sample);
            DoubleMatrix delta_vbias = L_vbias.columnSums().divi(nbr_sample);

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
