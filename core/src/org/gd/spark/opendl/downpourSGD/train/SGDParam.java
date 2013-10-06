package org.gd.spark.opendl.downpourSGD.train;

import java.io.Serializable;

/**
 * SGD based parameter
 * 
 * @author GuoDing
 * @since 2013-10-05
 */
public abstract class SGDParam implements Serializable {
	private static final long serialVersionUID = 1L;

	/**
	 * SGD parameter duplicate
	 * @return New parameter copy
	 */
	public abstract SGDParam dup();
}
