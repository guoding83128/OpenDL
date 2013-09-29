package org.gd.spark.opendl.downpourSGD.DeepNetwork;

import org.gd.spark.opendl.downpourSGD.hLayer.HiddenLayer;
import org.gd.spark.opendl.downpourSGD.hLayer.dA.dA;

public final class SdA extends DeepNetwork {
    private static final long serialVersionUID = 1L;

    public SdA(int _n_in, int _n_out, int[] _hidden_layer) {
        super(_n_in, _n_out, _hidden_layer);
    }

    public SdA(int _n_in, int _n_out, dA[] _da_layer) {
        super(_n_in, _n_out, _da_layer);
    }

    @Override
    protected HiddenLayer makeHiddenLayer(int n_visible, int n_hidden) {
        return new dA(n_visible, n_hidden);
    }
}
