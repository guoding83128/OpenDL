package org.spark.opendl.downpourSGD.lr;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.io.Writer;

import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.spark.opendl.downpourSGD.SGDPersistable;
import org.spark.opendl.downpourSGD.SGDTrainConfig;
import org.spark.opendl.util.MyConjugateGradient;

import cc.mallet.optimize.Optimizable;

/**
 * Logistic Regression(Softmax) node <p/>
 * 
 * @author GuoDing
 * @since 2013-08-01
 */
public final class LR implements SGDPersistable, Serializable {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(LR.class);
    private int x_num;
    private int y_num;
    private DoubleMatrix w;
    private DoubleMatrix b;

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
        if (null == _w) {
            w = new DoubleMatrix(y_num, x_num);
        } else {
            w = new DoubleMatrix(_w);
        }
        if (null == _b) {
            b = new DoubleMatrix(y_num);
        } else {
            b = new DoubleMatrix(_b);
        }
    }

    /**
     * Do predict work with multiple sample
     * 
     * @param x Input samples matrix
     * @return Predict result matrix (row=x's row, column=class num)
     */
    public final DoubleMatrix predict(DoubleMatrix x) {
        DoubleMatrix y = x.mmul(w.transpose()).addiRowVector(b);
        softmax(y);
        return y;
    }

    /**
     * Do predict work with one sample
     * 
     * @param x Input sample
     * @param y Output predict result
     */
    public final void predict(double[] x, double[] y) {
        for (int i = 0; i < y_num; i++) {
            y[i] = 0;
            for (int j = 0; j < x_num; j++) {
                y[i] += w.get(i, j) * x[j];
            }
            y[i] += b.get(i, 0);
        }
        softmax(y);
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
        return w;
    }

    /**
     * 
     * @return B param vector
     */
    public DoubleMatrix getB() {
        return b;
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

    /**
     * Gradient descent with mini-batch
     * 
     * @param config Train config
     * @param x_samples Input train samples X batch
     * @param y_samples Input train samples Y batch
     * @param curr_w W param matrix of current epoch
     * @param curr_b B param vector of current epoch
     */
    protected final void gradientUpdateMiniBatch(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, DoubleMatrix curr_w, DoubleMatrix curr_b) {
    	int nbr_samples = x_samples.getRows();
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

    /**
     * Conjugate gradient batch update
     * 
     * @param config Train config
     * @param x_samples Input train samples X batch
     * @param y_samples Input train samples Y batch
     * @param curr_w W param matrix of current epoch
     * @param curr_b B param vector of current epoch
     */
    protected final void gradientUpdateCG(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples,
            DoubleMatrix curr_w, DoubleMatrix curr_b) {
        LROptimizer lropt = new LROptimizer(config, x_samples, y_samples, curr_w, curr_b);
        MyConjugateGradient cg = new MyConjugateGradient(lropt, config.getCgInitStepSize());
        cg.setTolerance(config.getCgTolerance());
        try {
            cg.optimize(config.getCgMaxIterations());
        } catch (Throwable e) {
            logger.error("", e);
        } 
    }

    /**
     * Merge param update with one model replica
     * @param new_w W param matrix update
     * @param new_b B param vector update
     * @param nbr_model Number of model replica
     */
    protected final void mergeParam(DoubleMatrix new_w, DoubleMatrix new_b, int nbr_model) {
        w.addi(new_w.sub(w).divi(nbr_model));
        b.addi(new_b.sub(b).divi(nbr_model));
    }

    @Override
    public final void read(DataInput in) throws IOException {
        x_num = in.readInt();
        y_num = in.readInt();
        for (int i = 0; i < y_num; i++) {
            for (int j = 0; j < x_num; j++) {
                w.put(i, j, in.readDouble());
            }
        }
        for (int i = 0; i < y_num; i++) {
            b.put(i, 0, in.readDouble());
        }
    }

    @Override
    public final void write(DataOutput out) throws IOException {
        out.writeInt(x_num);
        out.writeInt(y_num);
        for (int i = 0; i < y_num; i++) {
            for (int j = 0; j < x_num; j++) {
                out.writeDouble(w.get(i, i));
            }
        }
        for (int i = 0; i < y_num; i++) {
            out.writeDouble(b.get(i, 0));
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
                wr.write(String.valueOf(w.get(i, i)));
                wr.write(",");
            }
            wr.write(newLine);
        }
        for (int i = 0; i < y_num; i++) {
            wr.write(String.valueOf(b.get(i, 0)));
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
}
