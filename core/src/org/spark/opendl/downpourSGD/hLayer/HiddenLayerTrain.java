package org.spark.opendl.downpourSGD.hLayer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;
import org.spark.opendl.downpourSGD.ModelReplicaSplit;
import org.spark.opendl.downpourSGD.SGDTrainConfig;
import org.spark.opendl.downpourSGD.SampleVector;

import spark.api.java.JavaPairRDD;

public final class HiddenLayerTrain implements Serializable {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(HiddenLayerTrain.class);

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

    public static void train(HiddenLayer hLayer, JavaPairRDD<Integer, SampleVector> sampleX, SGDTrainConfig config,
            boolean cache) {}
}
