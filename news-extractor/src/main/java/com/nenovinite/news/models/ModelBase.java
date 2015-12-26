package com.nenovinite.news.models;

import java.io.Serializable;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

import com.nenovinite.news.metrics.ModelEvaluation;

import scala.Tuple2;

public abstract class ModelBase implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3450876770200324012L;
		
	public abstract double getPreciction(Vector features);

	public JavaRDD<Tuple2<Object, Object>> getScoreAndLabels(JavaRDD<LabeledPoint> test) {
		// Compute raw scores on the test set.
		JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(p -> {
			Double score = this.getPreciction(p.features());
			return new Tuple2<Object, Object>(score, p.label());
		});

		return scoreAndLabels;
	};

	public void evaluate(JavaRDD<LabeledPoint> test) {
		JavaRDD<Tuple2<Object, Object>> scoreAndLabels = this.getScoreAndLabels(test).cache();

		final long testCount = test.count();
		ModelEvaluation.summary(scoreAndLabels, testCount);
		scoreAndLabels.unpersist();
	}
}
