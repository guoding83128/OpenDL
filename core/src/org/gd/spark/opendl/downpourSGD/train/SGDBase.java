package org.gd.spark.opendl.downpourSGD.train;

import java.io.Serializable;
import java.util.List;

import org.gd.spark.opendl.downpourSGD.SGDPersistable;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.jblas.DoubleMatrix;

/**
 * Stochastic Gradient Descent(SGD) based algorithm for the downpourSGD train framework
 * 
 * @author GuoDing
 * @since 2013-10-05
 */
public abstract class SGDBase implements SGDPersistable, Serializable {
	private static final long serialVersionUID = 1L;

	protected SGDParam param;
	
	/**
	 * Gradient descent with mini-batch<p/>
	 * @param config Train configuration
	 * @param x_samples X train data matrix
	 * @param y_samples Y train data matrix, will be null for unsupervise
	 * @param curr_param Parameter of current epoch
	 */
	public abstract void gradientUpdateMiniBatch(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param);
	
	/**
	 * Conjugate gradient batch update<p/>
	 * @param config Train configuration
	 * @param x_samples X train data matrix
	 * @param y_samples Y train data matrix, will be null for unsupervise
	 * @param curr_param Parameter of current epoch
	 */
	public abstract void gradientUpdateCG(SGDTrainConfig config, DoubleMatrix x_samples, DoubleMatrix y_samples, SGDParam curr_param);
	
	/**
	 * Merge param update with one model replica<p/>
     * Notice: use average merge w = w + (deltaw1 + deltaw2 + ... + deltawm)/m <p/>
	 * @param new_param New updated parameter
	 * @param nrModelReplica Number of model replica
	 */
	public abstract void mergeParam(SGDParam new_param, int nrModelReplica);
	
	/**
	 * Loss function calculation
	 * @param samples Input sample list
	 * @return Total loss for these samples
	 */
	public abstract double loss(List<SampleVector> samples);
	
	/**
	 * Whether it is both include X and Y data in sample
	 * @return
	 */
	public abstract boolean isSupervise();
	
	public final SGDParam getParam() {
		return param;
	}
}
