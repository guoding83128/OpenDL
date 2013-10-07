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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Writer;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.downpourSGD.train.SGDBase;
import org.gd.spark.opendl.downpourSGD.train.SGDParam;
import org.gd.spark.opendl.util.MathUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Hidden layer node framework <p/>
 * Notice, for dA and RBM, we use tied weights, so only one weight matrix need <p/>
 * 
 * @author GuoDing
 * @since 2013-07-15
 */
public abstract class HiddenLayer extends SGDBase {
    private static final long serialVersionUID = 1L;
    protected int n_visible;
    protected int n_hidden;
    protected HiddenLayerParam hlparam;

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
        hlparam = new HiddenLayerParam(_n_in, _n_out, _w, _b);
        param = hlparam;
    }

    /**
     * Input layer to hidden layer Sigmod output(standalone)
     * @param input Input layer matrix
     * @return Hidden layer output matrix
     */
    public final DoubleMatrix sigmod_output(DoubleMatrix input) {
        DoubleMatrix ret = input.mmul(hlparam.w.transpose()).addiRowVector(hlparam.hbias);
        MathUtil.sigmod(ret);
        return ret;
    }
    
    /**
     * Input layer to hidden layer Sigmod output
     * @param visible_x Input layer data
     * @param hidden_x Hidden layer output data
     */
    public final void sigmod_output(double[] visible_x, double[] hidden_x) {
    	for (int i = 0; i < n_hidden; i++) {
    		hidden_x[i] = 0;
            for (int j = 0; j < n_visible; j++) {
            	hidden_x[i] += hlparam.w.get(i, j) * visible_x[j];
            }
            hidden_x[i] += hlparam.hbias.get(i, 0);
            hidden_x[i] = MathUtil.sigmod(hidden_x[i]);
        }
    }
    
    /**
     * Input layer to hidden layer Sigmod output on Spark
     * @param samples Input layer data RDD
     * @param copyY Whether copy the y data from original input to output
     * @return Output sigmod data RDD
     */
    public final JavaRDD<SampleVector> sigmod_output(JavaRDD<SampleVector> samples, boolean copyY) {
    	return samples.map(new SigmodOutputSpark(copyY));
    }

    public DoubleMatrix getW() {
        return hlparam.w;
    }

    public DoubleMatrix getHBias() {
        return hlparam.hbias;
    }

    public DoubleMatrix getVBias() {
        return hlparam.vbias;
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
            	hlparam.w.put(i, j, in.readDouble());
            }
        }
        for (int i = 0; i < n_hidden; i++) {
        	hlparam.hbias.put(i, 0, in.readDouble());
        }
        for (int i = 0; i < n_visible; i++) {
        	hlparam.hbias.put(i, 0, in.readDouble());
        }
    }

    @Override
    public final void write(DataOutput out) throws IOException {
        out.writeInt(n_visible);
        out.writeInt(n_hidden);
        for (int i = 0; i < n_hidden; i++) {
            for (int j = 0; j < n_visible; j++) {
                out.writeDouble(hlparam.w.get(i, j));
            }
        }
        for (int i = 0; i < n_hidden; i++) {
            out.writeDouble(hlparam.hbias.get(i, 0));
        }
        for (int i = 0; i < n_visible; i++) {
            out.writeDouble(hlparam.vbias.get(i, 0));
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
                wr.write(String.valueOf(hlparam.w.get(i, j)));
                wr.write(",");
            }
            wr.write(newLine);
        }
        for (int i = 0; i < n_hidden; i++) {
            wr.write(String.valueOf(hlparam.hbias.get(i, 0)));
            wr.write(",");
        }
        wr.write(newLine);
        for (int i = 0; i < n_visible; i++) {
            wr.write(String.valueOf(hlparam.vbias.get(i, 0)));
            wr.write(",");
        }
        wr.write(newLine);
    }
    
    @Override
    protected final void mergeParam(SGDParam new_param, int nrModelReplica) {
    	HiddenLayerParam new_hlparam = (HiddenLayerParam)new_param;
    	hlparam.w.addi(new_hlparam.w.sub(hlparam.w).divi(nrModelReplica));
    	hlparam.hbias.addi(new_hlparam.hbias.sub(hlparam.hbias).divi(nrModelReplica));
    	hlparam.vbias.addi(new_hlparam.vbias.sub(hlparam.vbias).divi(nrModelReplica));
	}
    
    @Override
    protected final double loss(List<SampleVector> samples) {
    	DoubleMatrix x = MathUtil.convertX2Matrix(samples);
        DoubleMatrix reconstruct_x = reconstruct(x);
        return MatrixFunctions.powi(reconstruct_x.sub(x), 2).sum();
    }
    
    @Override
    protected final boolean isSupervise() {
		return false;
	}

    /**
     * Reconstruct process: from input layer to hidden layer<p/>
     * then convert back to visible layer with transpose W
     * @param input
     * @return
     */
    public abstract DoubleMatrix reconstruct(DoubleMatrix input);
    
    /**
     * Reconstruct process: from input layer to hidden layer<p/>
     * then convert back to visible layer with transpose W
     * @param x
     * @param reconstruct_x
     */
    public abstract void reconstruct(double[] x, double[] reconstruct_x);
    
    
    
    
    private class SigmodOutputSpark extends Function<SampleVector, SampleVector> {
		private static final long serialVersionUID = 1L;
		private boolean copyY = false;
		public SigmodOutputSpark(boolean _copyY) {
			copyY = _copyY;
		}
		@Override
		public SampleVector call(SampleVector arg) throws Exception {
			SampleVector ret = null;
			if(copyY) {
				ret = new SampleVector(n_hidden, arg.getY().length);
				for(int i = 0; i < arg.getY().length; i++) {
					ret.getY()[i] = arg.getY()[i];
				}
			}
			else {
				ret = new SampleVector(n_hidden);
			}
			sigmod_output(arg.getX(), ret.getX());
			return ret;
		}
    }
}
