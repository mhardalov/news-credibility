package com.nenovinite.news;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.cli.ParseException;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.Normalizer;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.optimization.L1Updater;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.codehaus.jettison.json.JSONException;
import org.codehaus.jettison.json.JSONObject;

import com.google.common.collect.Multiset;
import com.nenovinite.news.configuration.NewsConfiguration;
import com.nenovinite.news.features.TFIDFTransform;
import com.nenovinite.news.features.TokenTransform;
import com.nenovinite.news.models.ModelBase;

public class Main {

	private static DataFrame getBodyContent(SQLContext sqlContxt, String jsonPath, String bodyColumn,
			String whereClause, double label) {
		DataFrame df = sqlContxt.read().json(jsonPath);
		df.registerTempTable("news");
		df.printSchema();
		
		String sql = "SELECT " + bodyColumn + " as content, CAST(" + label + " as DOUBLE) as label FROM news "
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
			"не!новините", "\"дневник\"" }));


	public static void main(String[] args) throws ParseException {
		NewsConfiguration conf = new NewsConfiguration(args);

		SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("Parser news");
		try (JavaSparkContext sc = new JavaSparkContext(sparkConf)) {
			SQLContext sqlContxt = new SQLContext(sc);

			long seed = 11l;
			double[] weights = new double[] { 0.6, 0.4 };

			// Split initial RDD into two... [60% training data, 40% testing
			// data].
			DataFrame[] unreliableData = getBodyContent(sqlContxt, conf.getUnreliableDataset(), "content",
					" WHERE category = \"Политика\" ", 0.0).randomSplit(weights, seed);

			// " LIMIT 15000"
			DataFrame[] credibleData = getBodyContent(sqlContxt, conf.getCredibleDataset(), "BodyText", " LIMIT 35000",
					1.0).randomSplit(weights, seed);
			
			LuceneTokenizer tokenizer = new LuceneTokenizer()
					.setInputCol("content")
					.setOutputCol("tokens");
			
			StopWordsRemover remover = new StopWordsRemover()
					.setCaseSensitive(false)
					.setStopWords(STOP_WORDS.toArray(new String[STOP_WORDS.size()]))
					.setInputCol("tokens")
					.setOutputCol("filtered");
			
			NGram ngramTransformer = new NGram()
					.setInputCol("filtered")
					.setOutputCol("ngrams")
					.setN(3);
			
			int numFeatures = 2000;
			HashingTF hashingTF = new HashingTF()
					.setInputCol("ngrams")
					.setOutputCol("tf")
					.setNumFeatures(numFeatures);
			
			IDF idf = new IDF()
					.setInputCol("tf")
					.setOutputCol("idf")
					.setMinDocFreq(10);
			
			VectorAssembler assembler = new VectorAssembler()
					  .setInputCols(new String[]{"idf"})
					  .setOutputCol("features");
							
			Pipeline pipeline = new Pipeline()
					  .setStages(new PipelineStage[] {tokenizer, remover, ngramTransformer, hashingTF, idf, assembler});
			PipelineModel model = pipeline.fit(unreliableData[0]);
			
			unreliableData[0] = model.transform(unreliableData[0]);

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

			if (1 == 1) {
				return;
			}

			TokenTransform tokenizer = new TokenTransform(conf.isVerbose());

			// Random shuffle by sort by content
			JavaPairRDD<Double, Multiset<String>> trainingDocs = unreliableData[0].unionAll(credibleData[0])
					.sort("content").javaRDD().mapToPair(tokenizer::transform).cache();
			trainingDocs.cache();

			JavaPairRDD<Double, Multiset<String>> testDocs = unreliableData[1].unionAll(credibleData[1]).sort("content")
					.javaRDD().mapToPair(tokenizer::transform);
			testDocs.cache();

			long trainingCount = trainingDocs.count();

			TFIDFTransform tfIdf = new TFIDFTransform(trainingCount, conf.isVerbose());
			tfIdf.extract(trainingDocs);

			JavaRDD<LabeledPoint> training = trainingDocs.map(tfIdf::transform);
			training.cache();
			trainingDocs.unpersist();
			StandardScalerModel scaler = new StandardScaler().fit(training.map(row -> row.features()).rdd());
			training = training.map(row -> new LabeledPoint(row.label(), scaler.transform(row.features())));

			JavaRDD<LabeledPoint> test = testDocs.map(tfIdf::transform);
			test = test.map(row -> new LabeledPoint(row.label(), scaler.transform(row.features())));
			test.cache();
			testDocs.unpersist();

			ModelBase model = conf.getModel(training);
			training.unpersist();

			test.cache();

			long unreliableCount = testDocs.filter(row -> row._1 == 0).count();
			long credibleCount = testDocs.filter(row -> row._1 == 1).count();

			Accumulator<Integer> counterFor0 = sc.accumulator(0);
			Accumulator<Integer> corrcetFor0 = sc.accumulator(0);
			String output = model.evaluate(test, counterFor0, corrcetFor0);

			test.unpersist();

			System.out.println(output);
			System.out.println("Classified as Ne!Novinite: " + counterFor0.value());
			System.out.println("Correct Ne!Novinite: " + corrcetFor0.value());
			System.out.println("Features count:" + tfIdf.getFeaturesCount());

			System.out.println("Ne!Novite news:" + unreliableCount);
			System.out.println("Dnevnik news:" + credibleCount);
		}
	}

}
