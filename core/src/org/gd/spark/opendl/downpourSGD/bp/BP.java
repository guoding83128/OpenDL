package org.gd.spark.opendl.downpourSGD.bp;

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
import org.jblas.DoubleMatrix;

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
	
	private int n_in;
	private int n_out;
	private int[] n_hiddens;
	private BPParam bpparam;
	
	public BP(int _in, int _out, int[] _hiddens) {
		this(_in, _out, _hiddens, null, null);
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
	public void gradientUpdateMiniBatch(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param) {
	}

	@Override
	public void gradientUpdateCG(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param) {
	}

	@Override
	public void mergeParam(SGDParam new_param, int nrModelReplica) {
		BPParam new_bpparam = (BPParam)new_param;
		for(int i = 0; i < bpparam.w.length; i++) {
			bpparam.w[i].addi(new_bpparam.w[i].sub(bpparam.w[i]).divi(nrModelReplica));
		}
		for(int i = 0; i < bpparam.b.length; i++) {
			bpparam.b[i].addi(new_bpparam.b[i].sub(bpparam.b[i]).divi(nrModelReplica));
		}
	}

	@Override
	public double loss(List<SampleVector> samples) {
		return 0;
	}

	@Override
	public boolean isSupervise() {
		return true;
	}

}
