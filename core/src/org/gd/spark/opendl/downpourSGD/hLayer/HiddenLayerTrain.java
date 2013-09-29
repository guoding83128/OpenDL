package org.gd.spark.opendl.downpourSGD.hLayer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;
import org.gd.spark.opendl.downpourSGD.ModelReplicaSplit;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;

import spark.api.java.JavaPairRDD;
import spark.api.java.JavaRDD;

/**
 * Hidden layer train work api <p/>
 * 
 * @author GuoDing
 * @since 2013-08-18
 */
public final class HiddenLayerTrain implements Serializable {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(HiddenLayerTrain.class);

    /**
     * Hidden layer train work with multiple thread standalone
     * @param hLayer The hidden layer object to be trained
     * @param sampleX Input sample data
     * @param config Train configuration
     */
    public static void train(HiddenLayer hLayer, List<SampleVector> sampleX, SGDTrainConfig config) {
        int x_n = (int) sampleX.size();
        int nrModelReplica = config.getNbrModelReplica();
        HashMap<Integer, List<SampleVector>> list_map = new HashMap<Integer, List<SampleVector>>();
        DeltaThread[] threads = new DeltaThread[nrModelReplica];
        LossThread[] loss_threads = new LossThread[nrModelReplica];
        for (int i = 0; i < nrModelReplica; i++) {
            threads[i] = new DeltaThread(hLayer, config);
            loss_threads[i] = new LossThread(hLayer);
            list_map.put(i, new ArrayList<SampleVector>());
        }

        Random rand = new Random(System.currentTimeMillis());
        for (SampleVector v: sampleX) {
            int id = rand.nextInt(nrModelReplica);
            list_map.get(id).add(v);
        }

        // start iteration
        for (int epoch = 1; epoch <= config.getMaxEpochs(); epoch++) {
            // thread start
            for (int i = 0; i < nrModelReplica; i++) {
                threads[i].train(list_map.get(i), epoch);
            }

            // waiting for all stop
            while (true) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    break;
                }
                boolean allStop = true;
                for (int i = 0; i < nrModelReplica; i++) {
                    if (threads[i].isRunning()) {
                        allStop = false;
                        break;
                    }
                }
                if (allStop) {
                    break;
                }
            }

            // update param
            for (int i = 0; i < nrModelReplica; i++) {
                hLayer.mergeParam(threads[i].getW(), threads[i].getHbias(), threads[i].getVbias(), nrModelReplica);
            }

            logger.info("update done for this iteration-" + epoch);

            if (0 != (epoch % config.getLossCalStep())) {
                continue;
            }

            // sum loss
            for (int i = 0; i < nrModelReplica; i++) {
                loss_threads[i].sumLoss(threads[i].getSamples());
            }

            // waiting for all stop
            while (true) {
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    break;
                }
                boolean allStop = true;
                for (int i = 0; i < nrModelReplica; i++) {
                    if (loss_threads[i].isRunning()) {
                        allStop = false;
                        break;
                    }
                }
                if (allStop) {
                    break;
                }
            }

            // sum up
            double totalError = 0;
            for (int i = 0; i < nrModelReplica; i++) {
                totalError += loss_threads[i].getError();
            }
            totalError /= x_n;
            logger.info("iteration-" + epoch + " done, total error is " + totalError);
            if (totalError <= config.getMinLoss()) {
                break;
            }
        }
    }

    /**
     * Hidden layer train work with Spark platform
     * @param hLayer The hidden layer object to be trained
     * @param sampleX Input sample data
     * @param config Train configuration
     * @param cache If true, the sample data will be in Spark memory cached
     */
    public static void train(HiddenLayer hLayer, JavaRDD<SampleVector> sampleX, SGDTrainConfig config) {
    	int x_n = (int) sampleX.count();
        int nrModelReplica = config.getNbrModelReplica();
        
        // model split
        ModelReplicaSplit<SampleVector> split = new ModelReplicaSplit<SampleVector>();
        JavaPairRDD<Integer, List<SampleVector>> splitX = split.split(sampleX, nrModelReplica, config);
        logger.info("Model split done.");
        
        // start iteration
        for (int epoch = 1; epoch <= config.getMaxEpochs(); epoch++) {
        	JavaRDD<DeltaSpark> deltas = splitX.map(new DeltaSpark(hLayer, config, epoch));
        	for (DeltaSpark delta: deltas.collect()) {
        		hLayer.mergeParam(delta.getW(), delta.getHbias(), delta.getVbias(), nrModelReplica);
        	}
        	
        	logger.info("train done for this iteration-" + epoch);
        	
        	if (0 != (epoch % config.getLossCalStep())) {
                continue;
            }

            List<Double> loss_list = splitX.map(new LossSpark(hLayer)).collect();
            double error = 0;
            for (Double loss: loss_list) {
                error += loss;
            }
            error /= x_n;
            logger.info("iteration-" + epoch + " done, total error is " + error);
            if (error <= config.getMinLoss()) {
                break;
            }
        }
    }
}
