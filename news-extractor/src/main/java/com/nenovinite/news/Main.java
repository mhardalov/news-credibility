package com.nenovinite.news;

import java.security.SecureRandom;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.cli.ParseException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.SQLTransformer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;

import com.nenovinite.news.configuration.NewsConfiguration;


public class Main {

	private static DataFrame getBodyContent(SQLContext sqlContxt, String jsonPath, String bodyColumn,
			String whereClause, double label) {
		DataFrame df = sqlContxt.read().json(jsonPath);
		df.registerTempTable("news");
		df.printSchema();
		
		String sql = "SELECT generateId(" + bodyColumn + ") as id, " + bodyColumn + " as content, CAST(" + label + " as DOUBLE) as label FROM news "
				+ whereClause;
		DataFrame newsData = sqlContxt.sql(sql);
		return newsData;
	}

	
	public final static Set<String> STOP_WORDS = new HashSet<String>(Arrays.asList(new String[] { "а", "автентичен",
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


	public static void main(String[] args) throws ParseException {
		NewsConfiguration conf = new NewsConfiguration(args);
		
		final Random rand = new SecureRandom();

		SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("Parser news");
		try (JavaSparkContext sc = new JavaSparkContext(sparkConf)) {
			SQLContext sqlContxt = new SQLContext(sc);

			sqlContxt.udf().register("generateId", (String s) -> rand.nextInt(1000000), DataTypes.IntegerType);
			sqlContxt.udf().register("getFeatures", (String s) -> {
				int size = s.length();
				if (size > 0) {
					return size;
				}
				return size;
			}, DataTypes.IntegerType);
			
			long seed = 11l;
			double[] weights = new double[] { 0.7, 0.3 };

			// Split initial RDD into two... [60% training data, 40% testing
			// data].
			DataFrame[] unreliableData = getBodyContent(sqlContxt, conf.getUnreliableDataset(), "content",
					" WHERE category = \"Политика\" AND (content IS NOT NULL AND content <> '') ", 0.0).randomSplit(weights, seed);

			// " LIMIT 15000"
			DataFrame[] credibleData = getBodyContent(sqlContxt, conf.getCredibleDataset(), "BodyText", "WHERE (BodyText IS NOT NULL AND BodyText <> '') LIMIT 35000",
					1.0).randomSplit(weights, seed);
			
			DataFrame train = unreliableData[0].unionAll(credibleData[0]).orderBy("content").cache();
			DataFrame test = unreliableData[1].unionAll(credibleData[1]).orderBy("content").cache();			
			
			
			RegexTokenizer tokenizer = new RegexTokenizer()
					  .setInputCol("content")
					  .setOutputCol("tokens")
					  .setPattern("[\\s!,.?;'\"]+")
					  .setGaps(false);
			
			train = tokenizer.transform(train);
			
			TokenFeaturesExtractor tokenFeatures = new TokenFeaturesExtractor()
					.setInputCol(tokenizer.getOutputCol())
					.setOutputCol("tokenfeatures");
			train = tokenFeatures.transform(train);
			train.show();

			StopWordsRemover remover = new StopWordsRemover()
					.setCaseSensitive(false)
					.setStopWords(STOP_WORDS.toArray(new String[STOP_WORDS.size()]))
					.setInputCol(tokenizer.getOutputCol())
					.setOutputCol("filtered");
			
			NGram ngramTransformer = new NGram()
					.setInputCol(remover.getOutputCol())
					.setOutputCol("ngrams")
					.setN(3);
			
			int numFeatures = 2000;
			HashingTF hashingTF = new HashingTF()
					.setInputCol(ngramTransformer.getOutputCol())
					.setOutputCol("tf")
					.setNumFeatures(numFeatures);
			
			IDF idf = new IDF()
					.setInputCol(hashingTF.getOutputCol())
					.setOutputCol("idf")
					.setMinDocFreq(10);
			
			VectorAssembler assembler = new VectorAssembler()
					  .setInputCols(new String[]{idf.getOutputCol(), tokenFeatures.getOutputCol()})
					  .setOutputCol("features");
			
			LogisticRegression lr = new LogisticRegression()
					  .setMaxIter(10)
					  .setRegParam(0.01);
							
			Pipeline pipeline = new Pipeline()
					  .setStages(new PipelineStage[] {remover, ngramTransformer, hashingTF, idf, assembler, lr});
						
//			PipelineModel model = pipeline.fit(unreliableData[0]);
			
			// We use a ParamGridBuilder to construct a grid of parameters to search over.
			// With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
			// this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
			ParamMap[] paramGrid = new ParamGridBuilder()
					// 100, 1000
			    .addGrid(hashingTF.numFeatures(), new int[]{10, 100})
			    .addGrid(lr.regParam(), new double[]{0.1, 0.01})
			    .build();

			// We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
			// This will allow us to jointly choose parameters for all Pipeline stages.
			// A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
			// Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
			// is areaUnderROC.
			CrossValidator cv = new CrossValidator()
			  .setEstimator(pipeline)
			  .setEvaluator(new BinaryClassificationEvaluator())
			  .setEstimatorParamMaps(paramGrid)
			  .setNumFolds(2); // Use 3+ in practice

			// Run cross-validation, and choose the best set of parameters.
			CrossValidatorModel cvModel = cv.fit(train);
			
			// Make predictions on test documents. cvModel uses the best model found (lrModel).
			DataFrame predictions = cvModel.transform(test);
			// Select example rows to display.
			predictions.select("prediction", "label", "features").show(5);

			// Select (prediction, true label) and compute test error
			MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
			  .setLabelCol("label")
			  .setPredictionCol("prediction")
			  .setMetricName("precision");
			double accuracy = evaluator.evaluate(predictions);
			System.out.println("Test Error = " + (1.0 - accuracy));
			
//			unreliableData[0] = model.transform(unreliableData[0]);

//			unreliableData[0].javaRDD().mapToPair(v1 -> {
//				String text = v1.getString(0).replace("\n", " ").replace("\r", " ").replace("br2n", " ").replace("'", "\"");
//				Double label = v1.getDouble(1);
//				List<String> words = new ArrayList<>(Arrays.asList(text.split(" ")));
//				words.removeAll(Arrays.asList("", null, " "));
//				words.removeAll(STOP_WORDS);
//
//				return new Tuple2<>(label, words);
//			}).flatMapValues(words -> words).mapToPair(row -> {
//				return new Tuple2<>(row._1 + ",'" + row._2 + "'", 1);
//			}).reduceByKey((a, b) -> a + b).map(row -> row._1 + "," + row._2()).repartition(1)
//					.count();
//			saveAsTextFile("/home/momchil/Desktop/master-thesis/statistics/words.vectors");

//			TokenTransform tokenizer = new TokenTransform(conf.isVerbose());
//
//			// Random shuffle by sort by content
//			JavaPairRDD<Double, Multiset<String>> trainingDocs = unreliableData[0].unionAll(credibleData[0])
//					.sort("content").javaRDD().mapToPair(tokenizer::transform).cache();
//			trainingDocs.cache();
//
//			JavaPairRDD<Double, Multiset<String>> testDocs = unreliableData[1].unionAll(credibleData[1]).sort("content")
//					.javaRDD().mapToPair(tokenizer::transform);
//			testDocs.cache();
//
//			long trainingCount = trainingDocs.count();
//
//			TFIDFTransform tfIdf = new TFIDFTransform(trainingCount, conf.isVerbose());
//			tfIdf.extract(trainingDocs);
//
//			JavaRDD<LabeledPoint> training = trainingDocs.map(tfIdf::transform);
//			training.cache();
//			trainingDocs.unpersist();
//			StandardScalerModel scaler = new StandardScaler().fit(training.map(row -> row.features()).rdd());
//			training = training.map(row -> new LabeledPoint(row.label(), scaler.transform(row.features())));
//
//			JavaRDD<LabeledPoint> test = testDocs.map(tfIdf::transform);
//			test = test.map(row -> new LabeledPoint(row.label(), scaler.transform(row.features())));
//			test.cache();
//			testDocs.unpersist();
//
//			ModelBase model = conf.getModel(training);
//			training.unpersist();
//
//			test.cache();
//
//			long unreliableCount = testDocs.filter(row -> row._1 == 0).count();
//			long credibleCount = testDocs.filter(row -> row._1 == 1).count();
//
//			Accumulator<Integer> counterFor0 = sc.accumulator(0);
//			Accumulator<Integer> corrcetFor0 = sc.accumulator(0);
//			String output = model.evaluate(test, counterFor0, corrcetFor0);
//
//			test.unpersist();
//
//			System.out.println(output);
//			System.out.println("Classified as Ne!Novinite: " + counterFor0.value());
//			System.out.println("Correct Ne!Novinite: " + corrcetFor0.value());
//			System.out.println("Features count:" + tfIdf.getFeaturesCount());
//
//			System.out.println("Ne!Novite news:" + unreliableCount);
//			System.out.println("Dnevnik news:" + credibleCount);
		}
	}

}
