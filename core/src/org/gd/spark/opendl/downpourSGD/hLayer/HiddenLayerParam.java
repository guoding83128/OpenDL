package org.gd.spark.opendl.downpourSGD.hLayer;

import org.gd.spark.opendl.downpourSGD.train.SGDParam;
import org.gd.spark.opendl.util.MathUtil;
import org.jblas.DoubleMatrix;

class HiddenLayerParam extends SGDParam {
	private static final long serialVersionUID = 1L;
	protected DoubleMatrix w;
    protected DoubleMatrix hbias;
    protected DoubleMatrix vbias;
    
    public HiddenLayerParam(int _n_in, int _n_out) {
    	this(_n_in, _n_out, null, null);
    }
    
    public HiddenLayerParam(int _n_in, int _n_out, double[][] _w, double[] _b) {
    	int n_visible = _n_in;
    	int n_hidden = _n_out;
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
     * just for dup
     */
    private HiddenLayerParam() {
    }

	@Override
	public SGDParam dup() {
		HiddenLayerParam ret = new HiddenLayerParam();
		ret.w = w.dup();
		ret.hbias = hbias.dup();
		ret.vbias = vbias.dup();
		return ret;
	}

}
