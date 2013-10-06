package org.gd.spark.opendl.downpourSGD.bp;

import org.gd.spark.opendl.downpourSGD.train.SGDParam;
import org.gd.spark.opendl.util.MathUtil;
import org.jblas.DoubleMatrix;

class BPParam extends SGDParam {
	private static final long serialVersionUID = 1L;
    
	protected DoubleMatrix[] w;
	protected DoubleMatrix[] b;
	
	public BPParam(int _in, int _out, int[] _hiddens) {
		this(_in, _out, _hiddens, null, null);
	}
	
	public BPParam(int _in, int _out, int[] _hiddens, DoubleMatrix[] _w, DoubleMatrix[] _b) {
		int l = _hiddens.length + 1;
		w = new DoubleMatrix[l];
		b = new DoubleMatrix[l];
		
		for(int i = 0; i < l; i++) {
			int curr_in = -1;
			int curr_out = -1;
			if(0 == i) {
				curr_in = _in;
			}
			else {
				curr_in = _hiddens[i - 1];
			}
			if((l -1) == i) {
				curr_out = _out;
			}
			else {
				curr_out = _hiddens[i];
			}
			
			
			//w
			if((null != _w) && (null != _w[i])) {
				w[i] = _w[i].dup();
			}
			else {
				w[i] = new DoubleMatrix(curr_out, curr_in);
				double a = 1.0 / curr_in;
	            for (int j = 0; j < curr_out; j++) {
	                for (int k = 0; k < curr_in; k++) {
	                    w[i].put(j, k, MathUtil.uniform(-a, a));
	                }
	            }
			}
			
			//b
			if((null != _b) && (null != _b[i])) {
				b[i] = _b[i].dup();
			}
			else {
				b[i] = new DoubleMatrix(curr_out);
			}
		}
	}
	
	private BPParam() {
	}
	
	@Override
	public SGDParam dup() {
		BPParam ret = new BPParam();
		ret.w = new DoubleMatrix[w.length];
		for(int i = 0; i < w.length; i++) {
			ret.w[i] = w[i].dup();
		}
		ret.b = new DoubleMatrix[b.length];
		for(int i = 0; i < b.length; i++) {
			ret.b[i] = b[i].dup();
		}
		return ret;
	}

}
