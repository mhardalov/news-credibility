package com.nenovinite.news.features;

import java.io.IOException;
import java.io.Serializable;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.bg.BulgarianAnalyzer;
import org.apache.lucene.analysis.bg.BulgarianStemFilter;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.miscellaneous.LengthFilter;
import org.apache.lucene.analysis.shingle.ShingleFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.util.CharArraySet;
import org.apache.lucene.util.Attribute;
import org.apache.spark.sql.Row;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import com.nenovinite.news.features.filters.NumberFilter;

import scala.Tuple2;

public class TokenTransform implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4437649223110687193L;

	public static final CharArraySet BULGARIAN_STOP_WORDS_SET;

	private boolean verbose;

	public static final List<String> STOP_WORDS = Arrays.asList("а", "автентичен", "аз", "ако", "ала", "бе", "без",
			"беше", "би", "бивш", "бивша", "бившо", "бил", "била", "били", "било", "благодаря", "близо", "бъдат",
			"бъде", "бяха", "в", "вас", "ваш", "ваша", "вероятно", "вече", "взема", "ви", "вие", "винаги", "внимава",
			"време", "все", "всеки", "всички", "всичко", "всяка", "във", "въпреки", "върху", "г", "ги", "главен",
			"главна", "главно", "глас", "го", "година", "години", "годишен", "д", "да", "дали", "два", "двама",
			"двамата", "две", "двете", "ден", "днес", "дни", "до", "добра", "добре", "добро", "добър", "докато",
			"докога", "дори", "досега", "доста", "друг", "друга", "други", "е", "евтин", "едва", "един", "една",
			"еднаква", "еднакви", "еднакъв", "едно", "екип", "ето", "живот", "за", "забавям", "зад", "заедно", "заради",
			"засега", "заспал", "затова", "защо", "защото", "и", "из", "или", "им", "има", "имат", "иска", "й", "каза",
			"как", "каква", "какво", "както", "какъв", "като", "кога", "когато", "което", "които", "кой", "който",
			"колко", "която", "къде", "където", "към", "лесен", "лесно", "ли", "лош", "м", "май", "малко", "ме",
			"между", "мек", "мен", "месец", "ми", "много", "мнозина", "мога", "могат", "може", "мокър", "моля",
			"момента", "му", "н", "на", "над", "назад", "най", "направи", "напред", "например", "нас", "не", "него",
			"нещо", "нея", "ни", "ние", "никой", "нито", "нищо", "но", "нов", "нова", "нови", "новина", "някои",
			"някой", "няколко", "няма", "обаче", "около", "освен", "особено", "от", "отгоре", "отново", "още", "пак",
			"по", "повече", "повечето", "под", "поне", "поради", "после", "почти", "прави", "пред", "преди", "през",
			"при", "пък", "първата", "първи", "първо", "пъти", "равен", "равна", "с", "са", "сам", "само", "се", "сега",
			"си", "син", "скоро", "след", "следващ", "сме", "смях", "според", "сред", "срещу", "сте", "съм", "със",
			"също", "т", "тази", "така", "такива", "такъв", "там", "твой", "те", "тези", "ти", "т.н.", "то", "това",
			"тогава", "този", "той", "толкова", "точно", "три", "трябва", "тук", "тъй", "тя", "тях", "у", "утре",
			"харесва", "хиляди", "ч", "часа", "че", "често", "чрез", "ще", "щом", "юмрук", "я", "як", "zhelyo",
			"не!новините", "\"дневник\"", "br2n");

	static {

		final CharArraySet stopSet = new CharArraySet(STOP_WORDS, false);
		BULGARIAN_STOP_WORDS_SET = CharArraySet.unmodifiableSet(stopSet);
	}

	public TokenTransform(boolean verbose) {
		this.setVerbose(verbose);
	}

	public TokenTransform() {
		this(false);
	}

	public Tuple2<Double, Multiset<String>> transform(Row row) throws IOException {
		Double label = row.getDouble(1);
		StringReader document = new StringReader(row.getString(0).replaceAll("br2n", ""));
		List<String> wordsList = new ArrayList<>();

		try (BulgarianAnalyzer analyzer = new BulgarianAnalyzer()) {
			TokenStream stream = analyzer.tokenStream("words", document);

			TokenFilter lowerFilter = new LowerCaseFilter(stream);
			TokenFilter numbers = new NumberFilter(lowerFilter);
			// TokenFilter length = new LengthFilter(numbers, 3, 1000);
			TokenFilter stemmer = new BulgarianStemFilter(numbers);
			TokenFilter ngrams = new ShingleFilter(stemmer, 2, 3);

			try (TokenFilter filter = ngrams) {
				Attribute termAtt = filter.addAttribute(CharTermAttribute.class);
				filter.reset();
				while (filter.incrementToken()) {
					String word = termAtt.toString().replace(",", "(comma)").replaceAll("\n|\r", "");
					if (word.contains("_") || BULGARIAN_STOP_WORDS_SET.contains(word)) {
						continue;
					}
					wordsList.add(word);
				}
			}
		}

		Multiset<String> words = ConcurrentHashMultiset.create(wordsList);

		return new Tuple2<>(label, words);
	}

	public boolean isVerbose() {
		return verbose;
	}

	public void setVerbose(boolean verbose) {
		this.verbose = verbose;
	}

}
