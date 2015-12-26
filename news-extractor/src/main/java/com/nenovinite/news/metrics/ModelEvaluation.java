package com.nenovinite.news.metrics;

import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;

import scala.Tuple2;

public class ModelEvaluation {
	private ModelEvaluation() {
		
	}
	
	public static void summary(JavaRDD<Tuple2<Object, Object>> scoreAndLabels, final long testCount) {
		// Get evaluation metrics.
		BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
		long correct = scoreAndLabels.filter(pl -> pl._1().equals(pl._2())).count();

		double accuracy = (double) correct / testCount;

		// AUPRC
		double areaUnderPR = metrics.areaUnderPR();
		// AUROC
		double areaUnderROC = metrics.areaUnderROC();

		// Precision by threshold
		List<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD().collect();

		// Recall by threshold
		List<Tuple2<Object, Object>> recall = metrics.recallByThreshold().toJavaRDD().collect();

		// F Score by threshold
		List<Tuple2<Object, Object>> f1Score = metrics.fMeasureByThreshold().toJavaRDD().collect();

		List<Tuple2<Object, Object>> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD().collect();

		// Precision-recall curve
		List<Tuple2<Object, Object>> prc = metrics.pr().toJavaRDD().collect();

		// Thresholds
		List<Double> thresholds = metrics.precisionByThreshold().toJavaRDD().map(t -> new Double(t._1().toString()))
				.collect();

		// ROC Curve
		List<Tuple2<Object, Object>> roc = metrics.roc().toJavaRDD().collect();

		System.out.println("ROC curve: " + roc);
		System.out.println("Precision by threshold: " + precision);
		System.out.println("Recall by threshold: " + recall);
		System.out.println("F1 Score by threshold: " + f1Score);
		System.out.println("F2 Score by threshold: " + f2Score);
		System.out.println("Precision-recall curve: " + prc);
		System.out.println("Thresholds: " + thresholds);
		System.out.println("Area under precision-recall curve = " + areaUnderPR);
		System.out.println("Area under ROC = " + areaUnderROC);
		System.out.println("Accuracy: " + accuracy * 100);
	}

}
