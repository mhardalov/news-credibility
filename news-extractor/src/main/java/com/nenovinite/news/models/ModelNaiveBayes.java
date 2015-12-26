package com.nenovinite.news.models;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

public class ModelNaiveBayes extends ModelBase {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4945671256788878876L;
	final NaiveBayesModel model;

	public ModelNaiveBayes(JavaRDD<LabeledPoint> training) {
		model = NaiveBayes.train(training.rdd(), 1.0);
	}

	@Override
	public double getPreciction(Vector features) {
		return model.predict(features);
	}

}
