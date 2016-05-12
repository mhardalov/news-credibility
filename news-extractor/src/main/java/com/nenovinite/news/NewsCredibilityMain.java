package com.nenovinite.news;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.cli.ParseException;
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
	

	private static void evaluateModel(DataFrame df, Transformer model) {
		DataFrame predictions = predictForDF(df, model);

		// Select (prediction, true label) and compute test error
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
		  .setLabelCol("label")
		  .setPredictionCol("prediction")
		  .setMetricName("precision");
		double accuracy = evaluator.evaluate(predictions);
		
		// obtain metrics
		MulticlassMetrics metrics = new MulticlassMetrics(predictions.select("prediction", "label"));
//			StructField predictionColSchema = predictions.schema().apply("prediction");
		Integer numClasses = 2;//(Integer) MetadataUtils.getNumClasses(predictionColSchema).get();

		// compute the false positive rate per label
		StringBuilder results = new StringBuilder();
		results.append("label\tfpr\ttpr\trecall\tprecision\tfmeasure\n");
		for (int label = 0; label < numClasses; label++) {
		  results.append(label);
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

		Matrix confusionMatrix = metrics.confusionMatrix();
		// output the Confusion Matrix
		System.out.println("Confusion Matrix");
		System.out.println(confusionMatrix);
		System.out.println();
		System.out.println(results);
		System.out.println("F-measure\t" + metrics.fMeasure());
		System.out.println("Precision\t" + metrics.precision());
		System.out.println("Accuracy\t" + accuracy);
		System.out.println("Test-Error\t" + (1.0 - accuracy));
		System.out.println();
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
			assmeblerInput.add(idf.getOutputCol());
//			assmeblerInput.add(word2Vec.getOutputCol());
			assmeblerInput.add("commonfeatures");
		
		VectorAssembler assembler = new VectorAssembler()
				  .setInputCols(assmeblerInput.toArray(new String[assmeblerInput.size()]))
				  .setOutputCol("features");
		
		LogisticRegression lr = new LogisticRegression();
		
//			ngramTransformer, hashingTF, idf,
		Pipeline pipeline = new Pipeline()
				  .setStages(new PipelineStage[] { ngramTransformer, hashingTF, idf, /*word2Vec,*/  assembler, lr});
					
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

			double[] weights = new double[] { 0.7, 0.3 };

			DatasetLoader dataset = new DatasetLoader(sqlContxt, weights, conf);
			DataFrame train = dataset.getTrainingSet();
			DataFrame test = dataset.getTrainingSet();
			DataFrame validation = dataset.getValidationSet();
			
			Transformer model = trainModel(train, TOKENIZER_OUTPUT, false);
			
			System.out.println("Evaluation on training set\n");
			evaluateModel(train, model);
			
			System.out.println("Evaluation on testing set\n");
			evaluateModel(test, model);
			
			System.out.println("Evaluation on validation set\n");
			evaluateModel(validation, model);
		}
	}

}
