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
	protected SGDParam dup() {
		LRParam ret = new LRParam();
		ret.w = w.dup();
		ret.b = b.dup();
		return ret;
	}

}
