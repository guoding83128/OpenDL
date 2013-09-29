package org.gd.spark.opendl.downpourSGD.DeepNetwork;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.gd.spark.opendl.downpourSGD.SGDPersistable;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.downpourSGD.hLayer.HiddenLayer;
import org.gd.spark.opendl.downpourSGD.hLayer.HiddenLayerTrain;
import org.gd.spark.opendl.downpourSGD.lr.LR;
import org.gd.spark.opendl.downpourSGD.lr.LRTrain;
import org.jblas.DoubleMatrix;

import spark.api.java.JavaRDD;


/**
 * Deep networking (DBN, SdA) <p/>
 * You can organize the deep networking by yourself with dA, RBM.<p/>
 * The core is how to train each hidden layer in your deep-net 
 * 
 * @author GuoDing
 * @since 2013-09-14
 */
public abstract class DeepNetwork implements SGDPersistable, Serializable {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(DeepNetwork.class);
    protected int n_in;
    protected int n_out;
    protected int n_layers;
    protected int[] hidden_layer_sizes;
    protected HiddenLayer[] hidden_layer;
    protected LR lrNode;

    /**
     * Construct with all random initial hidden layer node
     * @param _n_in Input node number
     * @param _n_out Output node number
     * @param _hidden_layer Each hidden layer node number
     */
    public DeepNetwork(int _n_in, int _n_out, int[] _hidden_layer) {
        n_in = _n_in;
        n_out = _n_out;
        n_layers = _hidden_layer.length;
        hidden_layer_sizes = new int[n_layers];
        hidden_layer = new HiddenLayer[n_layers];

        /**
         * build network
         */
        int visible = 0;
        int hidden = 0;
        for (int i = 0; i < n_layers; i++) {
            hidden_layer_sizes[i] = _hidden_layer[i];
            hidden = hidden_layer_sizes[i];
            if (0 == i) {
                visible = n_in;
            } else {
                visible = hidden_layer_sizes[i - 1];
            }
            hidden_layer[i] = makeHiddenLayer(visible, hidden);
        }
        lrNode = new LR(hidden_layer_sizes[n_layers - 1], n_out);
    }

    /**
     * Construct with initial hidden layer node
     * @param _n_in Input node number
     * @param _n_out Output node number
     * @param _hidden_layer Each hidden layer node
     */
    public DeepNetwork(int _n_in, int _n_out, HiddenLayer[] _hidden_layer) {
        n_in = _n_in;
        n_out = _n_out;
        n_layers = _hidden_layer.length;
        hidden_layer_sizes = new int[n_layers];
        hidden_layer = new HiddenLayer[n_layers];

        for (int i = 0; i < n_layers; i++) {
            hidden_layer_sizes[i] = _hidden_layer[i].getHidden();
            hidden_layer[i] = _hidden_layer[i];
        }
        lrNode = new LR(hidden_layer_sizes[n_layers - 1], n_out);
    }

    /**
     * Produce the actual hidden layer node(dA or RBM)
     * @param n_visible
     * @param n_hidden
     * @return
     */
    protected abstract HiddenLayer makeHiddenLayer(int n_visible, int n_hidden);

    /**
     * Train the deep network with multiple thread standalone work <p/>
     * 1.train all the hidden layer with layer-wise
     * 2.train final output layer, the LogisticRegression(Softmax) node
     * @param samples Train samples
     * @param config Train configuration
     */
    public final void train(List<SampleVector> samples, SGDTrainConfig config) {
        List<SampleVector> curr_input = null;

        /**
         * layer-wise, train all hidden layer
         */
        for (int i = 0; i < n_layers; i++) {
            // get current layer input, forwarding
            if (0 == i) {
                curr_input = samples;
            } else {
                for (int j = 0; j < curr_input.size(); j++) {
                    double[] input = curr_input.get(j).getX();
                    SampleVector sample = new SampleVector(hidden_layer[i - 1].getHidden());
                    hidden_layer[i - 1].sigmod_output(input, sample.getX());
                    curr_input.set(j, sample);
                }
            }

            // train this layer
            HiddenLayerTrain.train(hidden_layer[i], curr_input, config);
            logger.info("hidden_layer[" + i + "] train work done.");
        }

        /**
         * lr train final
         */
        // get last hidden layer output
        List<SampleVector> xy_list = new ArrayList<SampleVector>();
        for (int j = 0; j < curr_input.size(); j++) {
            double[] input = curr_input.get(j).getX();
            double[] new_x = new double[hidden_layer[n_layers - 1].getHidden()];
            hidden_layer[n_layers - 1].sigmod_output(input, new_x);
            double[] y = samples.get(j).getY();

            SampleVector xy = new SampleVector(new_x, y);
            xy_list.add(xy);
        }

        LRTrain.train(lrNode, xy_list, config);
    }

    /**
     * Only train the final output layer
     * @param samples
     * @param config
     */
    public final void trainLR(List<SampleVector> samples, SGDTrainConfig config) {
        List<SampleVector> curr_input = new ArrayList<SampleVector>();
        for (SampleVector sample: samples) {
            curr_input.add(new SampleVector(sample.getX(), sample.getY()));
        }

        // sample forwarding
        for (int i = 0; i < n_layers; i++) {
            for (SampleVector sample: curr_input) {
                double[] input = sample.getX();
                double[] sample_output = new double[hidden_layer[i].getHidden()];
                hidden_layer[i].sigmod_output(input, sample_output);
                sample.setX(sample_output);
            }
        }

        // lr train
        LRTrain.train(lrNode, curr_input, config);
    }
    
    /**
     * Train the deep network with Spark framework <p/>
     * @param sampleX Input sample RDD data
     * @param config Train configuration
     */
    public final void train(JavaRDD<SampleVector> sampleX, SGDTrainConfig config) {
    }
    
    /**
     * Train the final output layer node with Spark framework <p/>
     * @param sampleX Input sample RDD data
     * @param config Train configuration
     */
    public final void trainLR(JavaRDD<SampleVector> sampleX, SGDTrainConfig config) {
    }
    
    /**
     * Do the prediction work with one sample
     * @param x Input sample
     * @param predict_y Output predict class data
     */
    public final void predict(double[] x, double[] predict_y) {
        double[] curr_input = x;

        // hidden layer forwarding
        for (int i = 0; i < n_layers; i++) {
            double[] curr_output = new double[hidden_layer[i].getHidden()];
            hidden_layer[i].sigmod_output(curr_input, curr_output);
            curr_input = curr_output;
        }

        // lr final
        lrNode.predict(curr_input, predict_y);
    }
    
    /**
     * Do the prediction work with batch data
     * @param x
     */
    public final DoubleMatrix predict(DoubleMatrix x) {
    	DoubleMatrix curr_input = x;
    	
    	// hidden layer forwarding
        for (int i = 0; i < n_layers; i++) {
        	curr_input = hidden_layer[i].sigmod_output(curr_input);
        }
        
        return lrNode.predict(curr_input);
    }

    @Override
    public void read(DataInput in) throws IOException {
        n_in = in.readInt();
        n_out = in.readInt();
        n_layers = in.readInt();
        for (int i = 0; i < n_layers; i++) {
            hidden_layer_sizes[i] = in.readInt();
        }
        for (int i = 0; i < n_layers; i++) {
            hidden_layer[i].read(in);
        }
        lrNode.read(in);
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(n_in);
        out.writeInt(n_out);
        out.writeInt(n_layers);
        for (int i = 0; i < n_layers; i++) {
            out.writeInt(hidden_layer_sizes[i]);
        }
        for (int i = 0; i < n_layers; i++) {
            hidden_layer[i].write(out);
        }
        lrNode.write(out);
    }

    @Override
    public void print(Writer wr) throws IOException {
        String newLine = System.getProperty("line.separator");
        wr.write(String.valueOf(n_in));
        wr.write(",");
        wr.write(String.valueOf(n_out));
        wr.write(",");
        wr.write(String.valueOf(n_layers));
        wr.write(newLine);
        for (int i = 0; i < n_layers; i++) {
            wr.write(String.valueOf(hidden_layer_sizes[i]));
            wr.write(",");
        }
        wr.write(newLine);
        for (int i = 0; i < n_layers; i++) {
            hidden_layer[i].print(wr);
            wr.write(newLine);
        }
        lrNode.print(wr);
        wr.write(newLine);
    }
}
