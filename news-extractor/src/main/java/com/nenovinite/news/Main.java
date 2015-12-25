package com.nenovinite.news;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang.math.NumberUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.spark.Accumulator;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
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
import org.apache.spark.util.StatCounter;
import org.codehaus.jettison.json.JSONException;
import org.codehaus.jettison.json.JSONObject;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;

import scala.Tuple2;

public class Main {
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

	public final static Analyzer analyzer = new StandardAnalyzer();

	private static void fixNews(String file) throws FileNotFoundException, IOException, JSONException {
		List<String> lines = new ArrayList<String>();
		int skipped = 0;
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			String line;
			while ((line = br.readLine()) != null) {
				try {
					JSONObject json = new JSONObject(line);
					lines.add(json.toString());
				} catch (Exception e) {
					skipped++;
				}
			}
		}

		System.out.println(skipped);
		FileWriter writer = new FileWriter(file + "-new");
		for (String str : lines) {
			writer.write(str + "\n");
		}
		writer.close();
	}

	public static void main(String[] args) {

		SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Parser news");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SQLContext sqlContxt = new SQLContext(sc);

		DataFrame newsData = getBodyContent(sqlContxt,
				"/home/momchil/Desktop/master-thesis/datasets/ne!novinite-data-extended.json", "content",
				" WHERE category = \"Политика\" ", 0.0).cache();
		long neNoviniteCount = newsData.count();
		newsData = newsData.unionAll(getBodyContent(sqlContxt,
				"/home/momchil/Desktop/master-thesis/datasets/Publications-all-2013-01-01-2015-04-01.json-new",
				"BodyText", " LIMIT 15000", 1.0)).sort("content").cache();
		final long allNewsCount = newsData.count();

		// Tokenizer tokenizer = new
		// Tokenizer().setInputCol("content").setOutputCol("words");
		// DataFrame wordsData = tokenizer.transform(newsData);
		//
		// int numFeatures = 5000;
		// HashingTF hashingTF = new
		// HashingTF().setInputCol("words").setOutputCol("rawFeatures")
		// .setNumFeatures(numFeatures);
		// DataFrame featurizedData = hashingTF.transform(wordsData);
		// IDF idf = new
		// IDF().setInputCol("rawFeatures").setOutputCol("features");
		// IDFModel idfModel = idf.fit(featurizedData);
		// DataFrame rescaledData = idfModel.transform(featurizedData);
		// JavaRDD<LabeledPoint> data = rescaledData.select("features",
		// "label").javaRDD()
		// .map(row -> new LabeledPoint(row.getDouble(1), row.getAs(0)));

		JavaPairRDD<Double, Multiset<String>> docs = newsData.javaRDD().mapToPair(v1 -> {
			Double label = v1.getDouble(1);
			List<String> wordsList = new ArrayList<>();

			try {
				try (TokenStream stream = analyzer.tokenStream("words", new StringReader(v1.getString(0)))) {
					stream.reset();
					while (stream.incrementToken()) {
						String word = stream.addAttribute(CharTermAttribute.class).toString();

						if (word.length() < 4 || NumberUtils.isNumber(word)) {
							continue;
						}

						wordsList.add(word);
					}
				}
			} catch (IOException e) {
				// not thrown b/c we're using a string reader...
				throw new RuntimeException(e);
			}
			wordsList.removeAll(STOP_WORDS);

			Multiset<String> words = ConcurrentHashMultiset.create(wordsList);

			return new Tuple2<>(label, words);
		}).cache();

		JavaPairRDD<String, Tuple2<Integer, Long>> idfRdd = docs.flatMap(row -> row._2())
				.mapToPair(word -> new Tuple2<>(word, 1)).partitionBy(new HashPartitioner(10))
				.reduceByKey((a, b) -> a + b).zipWithIndex()
				.mapToPair(row -> new Tuple2<>(row._1._1, new Tuple2<>(row._1._2, row._2))).cache();

		StatCounter idfStats = idfRdd.mapToDouble(row -> Math.log((double) (allNewsCount + 1) / (row._2()._1() + 1)))
				.stats();
		long featuresCount = idfStats.count();
		System.out.println(idfStats);
		for (Tuple2<String, Tuple2<Integer, Long>> word : idfRdd.takeOrdered(1000, new WordComparator())) {
			System.out.println(word._1() + "," + word._2._1);
		}
		Map<String, Tuple2<Integer, Long>> idf = idfRdd.collectAsMap();
		idfRdd.unpersist();

		JavaRDD<LabeledPoint> data = docs.map(doc -> {
			double label = doc._1();
			List<Tuple2<Integer, Double>> vector = new ArrayList<>();
			for (Multiset.Entry<String> entry : doc._2().entrySet()) {
				String word = entry.getElement();
				int tf = entry.getCount();

				Tuple2<Integer, Long> wordInfo = idf.get(word);
				int index = wordInfo._2().intValue();
				double idfScore = Math.log((double) (allNewsCount + 1) / (wordInfo._2() + 1));
				double tfidf = tf * idfScore;

				vector.add(new Tuple2<>(index, tfidf));
			}
			Vector features = Vectors.sparse((int)featuresCount, vector);

			return new LabeledPoint(label, features);
		});

		//
		// Split initial RDD into two... [60% training data, 40% testing data].
		JavaRDD<LabeledPoint> training = data.sample(false, 0.6, 11L);
		training.cache();
		JavaRDD<LabeledPoint> test = data.subtract(training);

		 evluateSVM(training, test);
//		evaluateNB(training, test, sc);
		System.out.println("Ne!Novite news:" + neNoviniteCount);
		System.out.println("Dnevnik news:" + (allNewsCount - neNoviniteCount));

		sc.close();

	}

	private static void evaluateNB(JavaRDD<LabeledPoint> training, JavaRDD<LabeledPoint> test, JavaSparkContext sc) {
		final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
		JavaPairRDD<Double, Double> predictionAndLabel = test
				.mapToPair(p -> new Tuple2<Double, Double>(model.predict(p.features()), p.label()));

		Accumulator<Integer> predictedNeNovinite = sc.accumulator(0);
		double accuracy = predictionAndLabel.filter(pl -> {
			if (pl._1() == 0.0) {
				predictedNeNovinite.add(1);
			}
			return pl._1().equals(pl._2());
		}).count() / (double) test.count();
		System.out.println(accuracy);
		System.out.println(predictedNeNovinite.value());

	}

	private static void evluateSVM(JavaRDD<LabeledPoint> training, JavaRDD<LabeledPoint> test) {
		SVMWithSGD svmAlg = new SVMWithSGD();
		svmAlg.optimizer().setNumIterations(200).setRegParam(0.1).setUpdater(new L1Updater());
		final SVMModel model = svmAlg.run(training.rdd());

		// Clear the default threshold.
		model.clearThreshold();

		// Compute raw scores on the test set.
		JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(p -> {
			Double score = model.predict(p.features());
			return new Tuple2<Object, Object>(score, p.label());
		});

		// Get evaluation metrics.
		BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));

		double accuracy = scoreAndLabels.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();
		double auROC = metrics.areaUnderROC();

		System.out.println("Area under ROC = " + auROC);
		System.out.println(accuracy);
	}

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

}
