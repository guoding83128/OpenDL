package org.spark.opendl.downpourSGD.DeepNetwork;

import org.spark.opendl.downpourSGD.hLayer.HiddenLayer;
import org.spark.opendl.downpourSGD.hLayer.RBM.RBM;

public final class DBN extends DeepNetwork {
    private static final long serialVersionUID = 1L;

    public DBN(int _n_in, int _n_out, int[] _hidden_layer) {
        super(_n_in, _n_out, _hidden_layer);
    }

    public DBN(int _n_in, int _n_out, RBM[] _rbm_layer) {
        super(_n_in, _n_out, _rbm_layer);
    }

    @Override
    protected HiddenLayer makeHiddenLayer(int n_visible, int n_hidden) {
        return new RBM(n_visible, n_hidden);
    }
}
