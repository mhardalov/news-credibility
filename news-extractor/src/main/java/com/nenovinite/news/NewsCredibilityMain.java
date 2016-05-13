package com.nenovinite.news;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.cli.ParseException;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

import com.nenovinite.news.configuration.NewsConfiguration;
import com.nenovinite.news.dataset.DatasetLoader;


public class NewsCredibilityMain {

	
	private static final String TSV_TEMPLATE = "/home/momchil/Documents/MasterThesis/experiments/%s.tsv";
	private static final String TOKENIZER_OUTPUT = "tokens";
	private final static Set<String> STOP_WORDS = new HashSet<String>(Arrays.asList(new String[] { "а", "автентичен",
			"аз", "ако", "ала", "бе", "без", "беше", "би", "бивш", "бивша", "бившо", "бил", "била", "били", "било",
			"благодаря", "близо", "бъдат", "бъде", "бяха", "в", "вас", "ваш", "ваша", "вероятно", "вече", "взема", "ви",
			"вие", "винаги", "внимава", "време", "все", "всеки", "всички", "всичко", "всяка", "във", "въпреки", "върху",
			"г", "ги", "главен", "главна", "главно", "глас", "го", "година", "години", "годишен", "д", "да", "дали",
			"два", "двама", "двамата", "две", "двете", "ден", "днес", "дни", "до", "добра", "добре", "добро", "добър",
			"докато", "докога", "дори", "досега", "доста", "друг", "друга", "други", "е", "евтин", "едва", "един",
			"една", "еднаква", "еднакви", "еднакъв", "едно", "екип", "ето", "живот", "за", "забавям", "зад", "заедно",
			"заради", "засега", "заспал", "затова", "защо", "защото", "и", "из", "или", "им", "има", "имат", "иска",
			"й", "каза", "как", "каква", "какво", "както", "какъв", "като", "кога", "когато", "което", "които", "кой",
			"който", "колко", "която", "къде", "където", "към", "лесен", "лесно", "ли", "лош", "м", "май", "малко",
			"ме", "между", "мек", "мен", "месец", "ми", "много", "мнозина", "мога", "могат", "може", "мокър", "моля",
			"момента", "му", "н", "на", "над", "назад", "най", "направи", "напред", "например", "нас", "не", "него",
			"нещо", "нея", "ни", "ние", "никой", "нито", "нищо", "но", "нов", "нова", "нови", "новина", "някои",
			"някой", "няколко", "няма", "обаче", "около", "освен", "особено", "от", "отгоре", "отново", "още", "пак",
			"по", "повече", "повечето", "под", "поне", "поради", "после", "почти", "прави", "пред", "преди", "през",
			"при", "пък", "първата", "първи", "първо", "пъти", "равен", "равна", "с", "са", "сам", "само", "се", "сега",
			"си", "син", "скоро", "след", "следващ", "сме", "смях", "според", "сред", "срещу", "сте", "съм", "със",
			"също", "т", "тази", "така", "такива", "такъв", "там", "твой", "те", "тези", "ти", "т.н.", "то", "това",
			"тогава", "този", "той", "толкова", "точно", "три", "трябва", "тук", "тъй", "тя", "тях", "у", "утре",
			"харесва", "хиляди", "ч", "часа", "че", "често", "чрез", "ще", "щом", "юмрук", "я", "як", "zhelyo",
			"не!новините", "\"дневник\"", "br2n" }));
	
	private static String stagesToString;
	
	private static double calcFMeasure(double precision, double recall, double beta) {
		double betaPow2 = Math.pow(beta, 2);
		double fMeasure = ((1 + betaPow2 ) * precision * recall) / (betaPow2 * (precision) + recall);
		return fMeasure;
	}

	private static void evaluateModel(SQLContext sqlContxt, DataFrame df, Transformer model, String outputPath, String title) {
		DataFrame predictions = predictForDF(df, model).select("prediction", "label").cache();

		// Select (prediction, true label) and compute test error
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
		  .setLabelCol("label")
		  .setPredictionCol("prediction")
		  .setMetricName("precision");
		double accuracy = evaluator.evaluate(predictions);
		
		// obtain metrics
		MulticlassMetrics metrics = new MulticlassMetrics(predictions);
		predictions.unpersist();
//			StructField predictionColSchema = predictions.schema().apply("prediction");
		Integer numClasses = 2;//(Integer) MetadataUtils.getNumClasses(predictionColSchema).get();

		// compute the false positive rate per label
		StringBuilder results = new StringBuilder();
		
		results.append(title + "\n");
		results.append("Features used:\t" + stagesToString + "\n");
		
		results.append("label\texamples\tfpr\ttpr\trecall\tprecision\tfmeasure\n");
		Matrix confusionMatrix = metrics.confusionMatrix();
		double tp = confusionMatrix.apply(0, 0);
		double fp = confusionMatrix.apply(1, 0);
		double fn = confusionMatrix.apply(0, 1);
		double tn = confusionMatrix.apply(1, 1);
		
		double[] examplesPerClass = new double[numClasses];
		for (int label = 0; label < numClasses; label++) {
			for (int i = 0; i < numClasses; i++) {
				examplesPerClass[label] += confusionMatrix.apply(label, i);
			}
		}
		
		for (int label = 0; label < numClasses; label++) {
		  results.append(label);
		  results.append("\t");
		  results.append(examplesPerClass[label]);
		  results.append("\t");
		  results.append(metrics.falsePositiveRate((double) label));
		  results.append("\t");
		  results.append(metrics.truePositiveRate((double) label));
		  results.append("\t");
		  results.append(metrics.recall((double) label));
		  results.append("\t");
		  results.append(metrics.precision((double) label));
		  results.append("\t");
		  results.append(metrics.fMeasure((double) label));
		  results.append("\n");
		}

		double precision = tp / (tp + fp);
		double recall = tp / (tp + fn);
		accuracy = (tp + tn) / (tp + fp + fn + tn);
		double f1Measure = calcFMeasure(precision, recall, 1.0);
		double f2Measure = calcFMeasure(precision, recall, 2.0);
		
		results.append("\n");
		
		// output the Confusion Matrix
		results.append("Confusion Matrix\n");
		results.append(confusionMatrix.toString().replaceAll("[ ]+", "\t") + "\n");
		results.append("\n");
		
		results.append("F-measure1\t" + f1Measure + "\n");
		results.append("F-measure2\t" + f2Measure + "\n");
		results.append("Precision\t" + precision + "\n");
		results.append("Recall\t" + recall + "\n");
		results.append("Accuracy\t" + accuracy  + "\n");
		results.append("Test-Error\t" + (1.0 - accuracy) + "\n");
		results.append("\n");
		
		System.out.println(results);
		
		try {
			(new File(outputPath)).createNewFile();
			Files.write(Paths.get(outputPath), results.toString().getBytes(), StandardOpenOption.APPEND);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}


	private static DataFrame predictForDF(DataFrame df, Transformer model) {
		df = getCommonFeatures(df, TOKENIZER_OUTPUT);

		// Make predictions on test documents. cvModel uses the best model found (lrModel).
		DataFrame predictions = model.transform(df);
		// Select example rows to display.
		predictions.select("prediction", "label", "features");
		return predictions;
	}


	private static Transformer trainModel(DataFrame train, String tokenizerOutputCol, boolean useCV) {
		train = getCommonFeatures(train, TOKENIZER_OUTPUT);
		
		NGram ngramTransformer = new NGram()
				.setInputCol("filtered")
				.setOutputCol("ngrams");
		
		HashingTF hashingTF = new HashingTF()
				.setInputCol(ngramTransformer.getOutputCol())
				.setOutputCol("tf");
		
		IDF idf = new IDF()
				.setInputCol(hashingTF.getOutputCol())
				.setOutputCol("idf");
		
		// Learn a mapping from words to Vectors.
		Word2Vec word2Vec = new Word2Vec()
		  .setInputCol(tokenizerOutputCol)
		  .setOutputCol("w2v");
		
		List<String> assmeblerInput = new ArrayList<>();
//			assmeblerInput.add(idf.getOutputCol());
			assmeblerInput.add(word2Vec.getOutputCol());
			assmeblerInput.add("commonfeatures");
		
		VectorAssembler assembler = new VectorAssembler()
				  .setInputCols(assmeblerInput.toArray(new String[assmeblerInput.size()]))
				  .setOutputCol("features");
		
		LogisticRegression lr = new LogisticRegression();
		
//			ngramTransformer, hashingTF, idf,
		PipelineStage[] pipelineStages = new PipelineStage[] { /*ngramTransformer, hashingTF, idf, */word2Vec,  assembler, lr};
		Pipeline pipeline = new Pipeline()
				  .setStages(pipelineStages);
		
		stagesToString = ("commonfeatures_suff1x\t" + StringUtils.join(pipelineStages, "\t")).replaceAll("([A-Za-z]+)_[0-9A-Za-z]+", "$1");
					
		// We use a ParamGridBuilder to construct a grid of parameters to search over.
		// With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
		// this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
		ParamMap[] paramGrid = new ParamGridBuilder()
//				.addGrid(word2Vec.vectorSize(), new int[] {100, 500})
//				.addGrid(word2Vec.minCount(), new int[] {2, 3, 4})
//				.addGrid(ngramTransformer.n(), new int[] {2, 3})
//				.addGrid(hashingTF.numFeatures(), new int[] {1000, 2000})
			.addGrid(lr.maxIter(), new int[] {10})
//		    .addGrid(lr.regParam(), new double[] {0.1, 0.001})
//		    .addGrid(lr.fitIntercept())
//			    .addGrid(idf.minDocFreq(), new int[]{2, 4})
		    .build();
		
		Transformer model;
		
		if (!useCV) {
			model = trainWithValidationSplit(train, pipeline, paramGrid);
		} else {
			model = trainWithCrossValidation(train, pipeline, paramGrid);
		}
		
		return model;
	}


	private static Transformer trainWithCrossValidation(DataFrame train, Pipeline pipeline, ParamMap[] paramGrid) {
		// We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
		// This will allow us to jointly choose parameters for all Pipeline stages.
		// A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
		// Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
		// is areaUnderROC.
		CrossValidator cv = new CrossValidator()
		  .setEstimator(pipeline)
		  .setEvaluator(new BinaryClassificationEvaluator())
		  .setEstimatorParamMaps(paramGrid)
		  .setNumFolds(10); // Use 3+ in practice

		// Run cross-validation, and choose the best set of parameters.
		CrossValidatorModel model = cv.fit(train);
		
		return model;
	}


	private static TrainValidationSplitModel trainWithValidationSplit(DataFrame train, Pipeline pipeline, ParamMap[] paramGrid) {
		// In this case the estimator is simply the linear regression.
		// A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
		TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
		  .setEstimator(pipeline)
		  .setEvaluator(new BinaryClassificationEvaluator())
		  .setEstimatorParamMaps(paramGrid)
		  .setTrainRatio(0.7); // 80% for training and the remaining 20% for validation

		// Run train validation split, and choose the best set of parameters.
		TrainValidationSplitModel model = trainValidationSplit.fit(train);
		return model;
	}


	private static DataFrame getCommonFeatures(DataFrame train, String tokenizerOutputCol) {
		RegexTokenizer tokenizer = new RegexTokenizer()
				  .setInputCol("content")
				  .setOutputCol(tokenizerOutputCol)
				  .setPattern("[\\s!,.?;'\"]+")
				  .setGaps(false);
		
		train = tokenizer.transform(train);
		
		TokenFeaturesExtractor tokenFeatures = new TokenFeaturesExtractor()
				.setInputCol(tokenizer.getOutputCol())
				.setOutputCol("commonfeatures");
		train = tokenFeatures.transform(train);
		
		StopWordsRemover remover = new StopWordsRemover()
				.setCaseSensitive(false)
				.setStopWords(STOP_WORDS.toArray(new String[STOP_WORDS.size()]))
				.setInputCol(tokenizer.getOutputCol())
				.setOutputCol("filtered");
		
		train = remover.transform(train);
		
		return train;
	}
	
	public static void main(String[] args) throws ParseException {
		final NewsConfiguration conf = new NewsConfiguration(args);

		SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("News Classificator");
		try (JavaSparkContext sc = new JavaSparkContext(sparkConf)) {
			SQLContext sqlContxt = new SQLContext(sc);

			Double[] weights = new Double[] { 0.7, 0.3 };
			List<Double> percents = Arrays.asList(weights).stream().mapToDouble(w -> w * 100.0).boxed().collect(Collectors.toList());

			DatasetLoader dataset = new DatasetLoader(sqlContxt, ArrayUtils.toPrimitive(weights), conf);
			DataFrame train = dataset.getTrainingSet();
			DataFrame test = dataset.getTestingSet();
			DataFrame validation = dataset.getValidationSet();
			
			Transformer model = trainModel(train, TOKENIZER_OUTPUT, false);

			String outputPath = prepareFile(TSV_TEMPLATE, percents);
			
			evaluateModel(sqlContxt, train, model, outputPath, "Evaluation on training set\n");
			evaluateModel(sqlContxt, test, model, outputPath, "Evaluation on testing set\n");
			evaluateModel(sqlContxt, validation, model, outputPath, "Evaluation on validation set\n");
		}
	}


	private static String prepareFile(String outputPath, List<Double> percents) {
		String fileName = "features:" + stagesToString.replace("\t", "_") + "_splits:" + StringUtils.join(percents, "_");
		outputPath = String.format(outputPath, fileName);
		(new File(outputPath)).delete();
		return outputPath;
	}

}
