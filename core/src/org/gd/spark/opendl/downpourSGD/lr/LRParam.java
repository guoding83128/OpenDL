package org.gd.spark.opendl.downpourSGD.lr;

import org.gd.spark.opendl.downpourSGD.train.SGDParam;
import org.jblas.DoubleMatrix;

class LRParam extends SGDParam {
	private static final long serialVersionUID = 1L;
	protected DoubleMatrix w;
	protected DoubleMatrix b;
    
    public LRParam(int x_feature_num, int y_class_num) {
    	this(x_feature_num, y_class_num, null, null);
    }
    
    public LRParam(int x_feature_num, int y_class_num, double[][] _w, double[] _b) {
    	if (null == _w) {
        	w = new DoubleMatrix(y_class_num, x_feature_num);
        } else {
            w = new DoubleMatrix(_w);
        }
        if (null == _b) {
            b = new DoubleMatrix(y_class_num);
        } else {
            b = new DoubleMatrix(_b);
        }
    }
    
    /**
     * just for dup
     */
    private LRParam() {
    }
    
	@Override
	public SGDParam dup() {
		LRParam ret = new LRParam();
		ret.w = w.dup();
		ret.b = b.dup();
		return ret;
	}

}
