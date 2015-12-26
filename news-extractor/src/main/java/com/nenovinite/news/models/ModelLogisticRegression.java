package com.nenovinite.news.models;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

public class ModelLogisticRegression extends ModelBase {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2533100807716870119L;
	
	final LogisticRegressionModel model;

	public ModelLogisticRegression(JavaRDD<LabeledPoint> training) {
		super();

		// Run training algorithm to build the model.
		model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training.rdd());

		// Clear the prediction threshold so the model will return probabilities
		model.clearThreshold();
	}

	@Override
	public double getPreciction(Vector features) {
		return model.predict(features);
	}

}
