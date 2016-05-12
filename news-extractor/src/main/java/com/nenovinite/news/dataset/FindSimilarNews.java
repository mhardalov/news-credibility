package com.nenovinite.news.dataset;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.cli.ParseException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import com.nenovinite.news.configuration.NewsConfiguration;

import scala.Tuple2;

public class FindSimilarNews {
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
			"не!новините", "\"дневник\"", "br2n", "ѝ", "zhelyobr2n"}));

	public static void main(String[] args) throws ParseException {
		final NewsConfiguration conf = new NewsConfiguration(args);
		
		SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("News Classificator");
		try (JavaSparkContext sc = new JavaSparkContext(sparkConf)) {
			SQLContext sqlContxt = new SQLContext(sc);
			double[] weights = new double[] { 0.7, 0.3 };

			DatasetLoader dataset = new DatasetLoader(sqlContxt, weights, conf);
			JavaRDD<Row> rowRdd = dataset.getUnreliableData()
//				.unionAll(dataset.getCredibleData())
				.selectExpr("content", "label")
				.javaRDD()
				.flatMapToPair(row -> {
					String content = row.getString(0).toLowerCase();
//					double label = row.getDouble(1);
					String[] wordsArr = content
						.replaceAll("[;.,!?\\(\\):'\"“”–\\\\]+", " ")
						.replaceAll("\\s+\\-\\s+", " ")
						.replaceAll("\\s+\\d+\\s+", " ")
						.replaceAll("br2n", " ")
						.replaceAll("\\s+", " ")
						.split(" ");
					
					List<String> words = new ArrayList<>(Arrays.asList(wordsArr));
					words.removeAll(STOP_WORDS);
					
					Map<String, Long> wordsMap = new HashMap<>();
					for (String word : words) {
						word = word.trim();
						if (word.isEmpty()) {
							continue;
						}
						Long count = wordsMap.get(word);
						if (count == null) {
							count = 0l;
						}
						count++;
						wordsMap.put(word, count);
					}
					
					List<Tuple2<String, Long>> document = wordsMap.entrySet()
						.stream()
						.parallel()
						.map(e -> new Tuple2<String, Long>(e.getKey(), e.getValue()))
						.collect(Collectors.toList());
					
					return document;
				})
				.reduceByKey((a, b) -> a + b)
				.map(r -> {
					return RowFactory.create(r._1, r._2);
				});
			
			// Generate the schema based on the string of schema
			StructType schema = DataTypes.createStructType(
					Arrays.asList(
							DataTypes.createStructField("word", DataTypes.StringType, true),
							DataTypes.createStructField("count", DataTypes.LongType, true)));
			
			sqlContxt.createDataFrame(rowRdd, schema)
				.registerTempTable("tfs");
			
			sqlContxt.sql("SELECT * FROM tfs WHERE word != '' ORDER BY count DESC  LIMIT 1").show();
			
//			String sqlText = "SELECT c_content, u_content\n" +
//							 "FROM unreliable\n"+
//							 "CROSS JOIN credible";
//			
//			DataFrame df = sqlContxt.sql(sqlText);
//			df.show();
//			System.out.println(df.count());
		}
	}

}
