package org.spark.opendl.downpourSGD;

import java.io.Serializable;

import lombok.Data;

/**
 * DownpourSGD train work configuration parameter <p/>
 * 
 * @author GuoDing
 * @since 2013-07-15
 */
@Data
public class SGDTrainConfig implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * number of model replica
     */
    private int nbrModelReplica = 1;

    /**
     * stop condition
     */
    private int maxEpochs = 500;
    private double minLoss = 0.1;

    /**
     * learning rate
     */
    public static final int LR_TYPE_FIX = 0;
    public static final int LR_TYPE_ADAGRAD = 1;
    public static final int LR_TYPE_CONFIG = 2;
    private int lrType = LR_TYPE_FIX;
    private double learningRate = 0.001; // only for fix lr, or adagrad basic lr
    private String lrPropertyName; // only for read lr from config file, specify the property name

    /**
     * regulation
     */
    private boolean useRegularization = false;
    private double lamada1 = 0; // most case, only use L2 reg is ok
    private double lamada2 = 0.0001;

    /**
     * parameter merge policy
     * 1.total merge w = w + deltaw1 + deltaw2 + ... + deltawm (m is the nbrModelReplica)
     * 2.average merge w = w + (deltaw1 + deltaw2 + ... + deltawm) / m
     */
    private boolean useAverageMerge = false;

    /**
     * caculate loss function
     */
    private int lossCalStep = 1; // if step <= 0, no loss cal, so only stop with maxEpochs.
    private boolean printLoss = true;

    /**
     * for dA
     */
    private boolean doCorruption = true;
    private double corruption_level = 0.3;

    /**
     * fro RBM
     */
    private int kStep = 1; // CD-k

    /**
     * mini batch gradient update when asynchronous SGD,
     * otherwise one by one.
     */
    private boolean useMiniBatch = false;

    /**
     * use ConjugateGradient on front step
     */
    // whether use cg on front step
    private boolean useCG = false;

    // stop use cg(switch to asynchronous SGD) after cgEpochStep epoch times.
    // default is 1, means only use cg first time(coverage fast to low loss).
    private int cgEpochStep = 1;

    // max iteration time in one time cg
    private int cgMaxIterations = 100;

    // the cg min gradient update tolerance
    private double cgTolerance = 0.0001;

    // cg init step size
    private double cgInitStepSize = 0.01;
}
