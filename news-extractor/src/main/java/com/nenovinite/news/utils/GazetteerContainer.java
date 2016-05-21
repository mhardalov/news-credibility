package com.nenovinite.news.utils;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.AbstractMap.SimpleEntry;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.Set;

import org.apache.lucene.analysis.util.CharArraySet;

public class GazetteerContainer {
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

	public static final Set<String> SINGULAR_PRONOUNS = new HashSet<>(
			Arrays.asList("аз", "ти", "той", "тя", "то", "мене", "мен", "ме", "мене", "ми", "тебе", "теб", "те", "тебе",
					"ти", "него", "нея", "него", "го", "я", "го", "нему", "ней", "нему", "му", "й", "му"));
	public static final Set<String> PLURAL_PRONOUNS = new HashSet<>(Arrays.asList("ние", "ний", "вие", "вий", "те",
			"нас", "ни", "нам", "ни", "вас", "ви", "вам", "ви", "тях", "ги", "тям", "им"));
	public static final Set<String> FIRST_PERSON = new HashSet<>(
			Arrays.asList("аз", "мене", "мен", "ме", "мене", "ми", "ние", "ний", "нас", "ни", "нам", "ни"));
	public static final Set<String> SECOND_PERSON = new HashSet<>(
			Arrays.asList("ти", "тебе", "теб", "те", "тебе", "ти", "вие", "вий", "вас", "ви", "вам", "ви"));
	public static final Set<String> THIRD_PERSON = new HashSet<>(Arrays.asList("тe", "тях", "ги", "тям", "им", "него",
			"нея", "него", "го", "я", "го", "нему", "ней", "нему", "му", "й", "му", "той", "тя", "то"));
	public static final Set<String> MONTHS = new HashSet<>(Arrays.asList("януари", "февруари", "март", "април", "май",
			"юни", "юли", "август", "септември", "октомври", "ноември", "декември"));

	public static final CharArraySet BULGARIAN_STOP_WORDS_SET;

	public static final Map<String, Double> POSITIVE_SENTIMENT;
	public static final Map<String, Double> NEGATIVE_SENTIMENT;

	private static Map<String, Double> readSentimentFiles(String file) {
		try {
			return Files.lines(Paths.get(file), Charset.forName("UTF-8")).map(line -> {
				String[] keyValue = line.split(",");
				Entry<String, Double> entry = new SimpleEntry<>(keyValue[0], Double.parseDouble(keyValue[1]));

				return entry;
			}).collect(Collectors.toMap(Entry::getKey, Entry::getValue, (e1, e2) -> e1, HashMap::new));
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return new HashMap<>();
	}

	static {
		POSITIVE_SENTIMENT = readSentimentFiles("/home/momchil/Documents/MasterThesis/features/sentiment/positive.txt");
		NEGATIVE_SENTIMENT = readSentimentFiles("/home/momchil/Documents/MasterThesis/features/sentiment/negative.txt");

		Set<String> stopWordsNew = new HashSet<>(STOP_WORDS);
		stopWordsNew.removeAll(SINGULAR_PRONOUNS);
		stopWordsNew.removeAll(PLURAL_PRONOUNS);

		final CharArraySet stopSet = new CharArraySet(stopWordsNew, true);
		BULGARIAN_STOP_WORDS_SET = CharArraySet.unmodifiableSet(stopSet);
	}
}
