package com.nenovinite.news.metrics;

import java.util.List;

import org.apache.commons.lang3.text.StrBuilder;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;

import scala.Tuple2;

public class ModelEvaluation {
	private ModelEvaluation() {
		
	}
	
	public static String summary(JavaRDD<Tuple2<Object, Object>> scoreAndLabels, final long testCount) {
		StrBuilder sb = new StrBuilder();
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

		
		sb.appendln("ROC curve: " + roc);
		sb.appendln("Precision by threshold: " + precision);
		sb.appendln("Recall by threshold: " + recall);
		sb.appendln("F1 Score by threshold: " + f1Score);
		sb.appendln("F2 Score by threshold: " + f2Score);
		sb.appendln("Precision-recall curve: " + prc);
		sb.appendln("Thresholds: " + thresholds);
		sb.appendln("Area under precision-recall curve = " + areaUnderPR);
		sb.appendln("Area under ROC = " + areaUnderROC);
		sb.appendln("Accuracy: " + accuracy * 100);
		
		return sb.toString();
	}

}
