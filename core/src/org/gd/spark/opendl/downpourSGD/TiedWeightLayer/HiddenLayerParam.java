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
package org.gd.spark.opendl.downpourSGD.TiedWeightLayer;

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
	protected SGDParam dup() {
		HiddenLayerParam ret = new HiddenLayerParam();
		ret.w = w.dup();
		ret.hbias = hbias.dup();
		ret.vbias = vbias.dup();
		return ret;
	}

}
