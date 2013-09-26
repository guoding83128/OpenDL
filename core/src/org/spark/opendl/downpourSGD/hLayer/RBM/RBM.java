package org.spark.opendl.downpourSGD.hLayer.RBM;

import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.spark.opendl.downpourSGD.SGDTrainConfig;
import org.spark.opendl.downpourSGD.hLayer.HiddenLayer;
import org.spark.opendl.downpourSGD.hLayer.HiddenLayerOptimizer;
import org.spark.opendl.util.MathUtil;
import org.spark.opendl.util.MyConjugateGradient;

public class RBM extends HiddenLayer {
    private static final Logger logger = Logger.getLogger(RBM.class);
    private static final long serialVersionUID = 1L;

    public RBM(int in_n_visible, int in_n_hidden) {
        super(in_n_visible, in_n_hidden);
    }

    public RBM(int in_n_visible, int in_n_hidden, double[][] in_w, double[] in_hbias, double[] in_vbias) {
        super(in_n_visible, in_n_hidden, in_w, in_hbias);
        if (null == in_vbias) {
            vbias = new DoubleMatrix(n_visible);
        } else {
            vbias = new DoubleMatrix(in_vbias);
        }
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
    protected void gradientUpdateMiniBatch(SGDTrainConfig config, DoubleMatrix samples, DoubleMatrix curr_w,
            DoubleMatrix curr_hbias, DoubleMatrix curr_vbias) {}

    @Override
    protected void gradientUpdateCG(SGDTrainConfig config, DoubleMatrix samples, DoubleMatrix curr_w,
            DoubleMatrix curr_hbias, DoubleMatrix curr_vbias) {
        RBMOptimizer rbmopt = new RBMOptimizer(config, samples, n_visible, n_hidden, curr_w, curr_hbias, curr_vbias);
        MyConjugateGradient cg = new MyConjugateGradient(rbmopt, config.getCgInitStepSize());
        cg.setTolerance(config.getCgTolerance());
        try {
            cg.optimize(config.getCgMaxIterations());
        } catch (Throwable e) {
            logger.error("", e);
        }
    }

    @Override
    protected DoubleMatrix reconstruct(DoubleMatrix input) {
        DoubleMatrix ret = input.mmul(w.transpose()).addiRowVector(hbias);
        MathUtil.sigmod(ret);
        ret = ret.mmul(w).addiRowVector(vbias);
        MathUtil.sigmod(ret);
        return ret;
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
            DoubleMatrix delta_w = ph_mean.transpose().mmul(my_samples).subi(nh_means.transpose().mmul(nv_samples))
                    .divi(nbr_sample);
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

	@Override
	protected void reconstruct(double[] x, double[] reconstruct_x) {
		// TODO Auto-generated method stub
		
	}
}
