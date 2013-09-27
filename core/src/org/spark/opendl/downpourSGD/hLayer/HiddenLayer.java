package org.spark.opendl.downpourSGD.hLayer;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.io.Writer;

import org.jblas.DoubleMatrix;
import org.spark.opendl.downpourSGD.SGDPersistable;
import org.spark.opendl.downpourSGD.SGDTrainConfig;
import org.spark.opendl.util.MathUtil;

/**
 * Hidden layer node framework <p/>
 * Notice, for dA and RBM, we use tied weights, so only one weight matrix need <p/>
 * 
 * @author GuoDing
 * @since 2013-07-15
 */
public abstract class HiddenLayer implements SGDPersistable, Serializable {
    private static final long serialVersionUID = 1L;
    protected int n_visible;
    protected int n_hidden;
    protected DoubleMatrix w;
    protected DoubleMatrix hbias;
    protected DoubleMatrix vbias;

    /**
     * Constructor with random initial parameters
     * 
     * @param _n_in The input node number
     * @param _n_out The hidden layer node number
     */
    public HiddenLayer(int _n_in, int _n_out) {
        this(_n_in, _n_out, null, null);
    }

    /**
     * Constructor with initial W matrix and hidden bias vector
     * 
     * @param _n_in The input node number
     * @param _n_out The hidden layer node number
     * @param _w W matrix
     * @param _b hidden bias vector
     */
    public HiddenLayer(int _n_in, int _n_out, double[][] _w, double[] _b) {
        n_visible = _n_in;
        n_hidden = _n_out;

        if (null == _w) {
            w = new DoubleMatrix(n_hidden, n_visible);
            double a = 1.0 / n_visible;
            for (int i = 0; i < n_hidden; i++) {
                for (int j = 0; j < n_visible; j++) {
                    w.put(i, j, MathUtil.uniform(-a, a));
                }
            }
        } else {
            w = new DoubleMatrix(_w);
        }

        if (null == _b) {
            this.hbias = new DoubleMatrix(n_hidden);
        } else {
            this.hbias = new DoubleMatrix(_b);
        }
        vbias = new DoubleMatrix(n_visible);
    }

    /**
     * Input layer to hidden layer Sigmod output
     * 
     * @param input Input layer matrix
     * @return Hidden layer output matrix
     */
    public final DoubleMatrix sigmod_output(DoubleMatrix input) {
        DoubleMatrix ret = input.mmul(w.transpose()).addiRowVector(hbias);
        MathUtil.sigmod(ret);
        return ret;
    }
    
    /**
     * Input layer to hidden layer Sigmod output
     * 
     * @param visible_x Input layer data
     * @param hidden_x Hidden layer output data
     */
    public final void sigmod_output(double[] visible_x, double[] hidden_x) {
    	for (int i = 0; i < n_hidden; i++) {
    		hidden_x[i] = 0;
            for (int j = 0; j < n_visible; j++) {
            	hidden_x[i] += w.get(i, j) * visible_x[j];
            }
            hidden_x[i] += hbias.get(i, 0);
            hidden_x[i] = MathUtil.sigmod(hidden_x[i]);
        }
    }

    public DoubleMatrix getW() {
        return w;
    }

    public DoubleMatrix getHBias() {
        return hbias;
    }

    public DoubleMatrix getVBias() {
        return vbias;
    }

    public int getVisible() {
        return n_visible;
    }

    public int getHidden() {
        return n_hidden;
    }

    @Override
    public final void read(DataInput in) throws IOException {
        n_visible = in.readInt();
        n_hidden = in.readInt();
        for (int i = 0; i < n_hidden; i++) {
            for (int j = 0; j < n_visible; j++) {
                w.put(i, j, in.readDouble());
            }
        }
        for (int i = 0; i < n_hidden; i++) {
            hbias.put(i, 0, in.readDouble());
        }
        for (int i = 0; i < n_visible; i++) {
            hbias.put(i, 0, in.readDouble());
        }
    }

    @Override
    public final void write(DataOutput out) throws IOException {
        out.writeInt(n_visible);
        out.writeInt(n_hidden);
        for (int i = 0; i < n_hidden; i++) {
            for (int j = 0; j < n_visible; j++) {
                out.writeDouble(w.get(i, j));
            }
        }
        for (int i = 0; i < n_hidden; i++) {
            out.writeDouble(hbias.get(i, 0));
        }
        for (int i = 0; i < n_visible; i++) {
            out.writeDouble(vbias.get(i, 0));
        }
    }

    @Override
    public final void print(Writer wr) throws IOException {
        String newLine = System.getProperty("line.separator");
        wr.write(String.valueOf(n_visible));
        wr.write(",");
        wr.write(String.valueOf(n_hidden));
        wr.write(newLine);
        for (int i = 0; i < n_hidden; i++) {
            for (int j = 0; j < n_visible; j++) {
                wr.write(String.valueOf(w.get(i, j)));
                wr.write(",");
            }
            wr.write(newLine);
        }
        for (int i = 0; i < n_hidden; i++) {
            wr.write(String.valueOf(hbias.get(i, 0)));
            wr.write(",");
        }
        wr.write(newLine);
        for (int i = 0; i < n_visible; i++) {
            wr.write(String.valueOf(vbias.get(i, 0)));
            wr.write(",");
        }
        wr.write(newLine);
    }

    /**
     * Merge param update with one model replica<p/>
     * Notice: use average merge w = w + (deltaw1 + deltaw2 + ... + deltawm)/m <p/>
     * @param new_w New updated weight matrix
     * @param new_hbias New updated hidden layer bias vector
     * @param new_vbias New updated visible layer bias vector
     * @param nbr_model Number of model replica
     */
    protected final void mergeParam(DoubleMatrix new_w, DoubleMatrix new_hbias, DoubleMatrix new_vbias, int nbr_model) {
        w.addi(new_w.sub(w).divi(nbr_model));
        hbias.addi(new_hbias.sub(hbias).divi(nbr_model));
        vbias.addi(new_vbias.sub(vbias).divi(nbr_model));
    }

    /**
     * Gradient descent with mini-batch
     * @param config Train config
     * @param samples Input samples
     * @param curr_w W matrix of current epoch
     * @param curr_hbias Hidden bias vector of current epoch
     * @param curr_vbias Visible bias matrix of current epoch
     */
    protected abstract void gradientUpdateMiniBatch(SGDTrainConfig config, DoubleMatrix samples, DoubleMatrix curr_w,
            DoubleMatrix curr_hbias, DoubleMatrix curr_vbias);

    /**
     * Conjugate gradient batch update
     * @param config Train config
     * @param samples Input samples
     * @param curr_w W matrix of current epoch
     * @param curr_hbias Hidden bias vector of current epoch
     * @param curr_vbias Visible bias matrix of current epoch
     */
    protected abstract void gradientUpdateCG(SGDTrainConfig config, DoubleMatrix samples, DoubleMatrix curr_w,
            DoubleMatrix curr_hbias, DoubleMatrix curr_vbias);

    /**
     * Reconstruct process: from input layer to hidden layer<p/>
     * then convert back to visible layer with transpose W
     * @param input
     * @return
     */
    protected abstract DoubleMatrix reconstruct(DoubleMatrix input);
    
    /**
     * Reconstruct process: from input layer to hidden layer<p/>
     * then convert back to visible layer with transpose W
     * @param x
     * @param reconstruct_x
     */
    protected abstract void reconstruct(double[] x, double[] reconstruct_x);
}
