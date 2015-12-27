package com.nenovinite.news.models;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.optimization.L1Updater;
import org.apache.spark.mllib.regression.LabeledPoint;

public class ModelSVM extends ModelBase {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3433215707803437881L;

	final SVMModel model;

	public ModelSVM(JavaRDD<LabeledPoint> training) {
		super();

		SVMWithSGD svmAlg = new SVMWithSGD();
		svmAlg.optimizer().setNumIterations(100).setRegParam(0.1).setUpdater(new L1Updater());
		model = svmAlg.run(training.rdd());

		// Clear the default threshold.
//		model.clearThreshold();
//		model.setThreshold(0.001338428);
	}

	@Override
	public double getPreciction(Vector features) {
		return model.predict(features);
	}
}
