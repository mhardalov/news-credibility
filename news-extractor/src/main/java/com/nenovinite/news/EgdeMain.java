package com.nenovinite.news;

import java.io.IOException;
import java.io.StringReader;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.math.NumberUtils;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.bg.BulgarianAnalyzer;
import org.apache.lucene.analysis.bg.BulgarianStemFilter;
import org.apache.lucene.analysis.core.LetterTokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.LowerCaseTokenizer;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.miscellaneous.LengthFilter;
import org.apache.lucene.analysis.shingle.ShingleFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.util.CharArraySet;
import org.apache.lucene.analysis.util.FilteringTokenFilter;
import org.apache.lucene.util.Attribute;

public class EgdeMain {

	public static final CharArraySet BULGARIAN_STOP_WORDS_SET;

	static {
		final List<String> stopWords = Arrays.asList("а", "автентичен", "аз", "ако", "ала", "бе", "без", "беше", "би",
				"бивш", "бивша", "бившо", "бил", "била", "били", "било", "благодаря", "близо", "бъдат", "бъде", "бяха",
				"в", "вас", "ваш", "ваша", "вероятно", "вече", "взема", "ви", "вие", "винаги", "внимава", "време",
				"все", "всеки", "всички", "всичко", "всяка", "във", "въпреки", "върху", "г", "ги", "главен", "главна",
				"главно", "глас", "го", "година", "години", "годишен", "д", "да", "дали", "два", "двама", "двамата",
				"две", "двете", "ден", "днес", "дни", "до", "добра", "добре", "добро", "добър", "докато", "докога",
				"дори", "досега", "доста", "друг", "друга", "други", "е", "евтин", "едва", "един", "една", "еднаква",
				"еднакви", "еднакъв", "едно", "екип", "ето", "живот", "за", "забавям", "зад", "заедно", "заради",
				"засега", "заспал", "затова", "защо", "защото", "и", "из", "или", "им", "има", "имат", "иска", "й",
				"каза", "как", "каква", "какво", "както", "какъв", "като", "кога", "когато", "което", "които", "кой",
				"който", "колко", "която", "къде", "където", "към", "лесен", "лесно", "ли", "лош", "м", "май", "малко",
				"ме", "между", "мек", "мен", "месец", "ми", "много", "мнозина", "мога", "могат", "може", "мокър",
				"моля", "момента", "му", "н", "на", "над", "назад", "най", "направи", "напред", "например", "нас", "не",
				"него", "нещо", "нея", "ни", "ние", "никой", "нито", "нищо", "но", "нов", "нова", "нови", "новина",
				"някои", "някой", "няколко", "няма", "обаче", "около", "освен", "особено", "от", "отгоре", "отново",
				"още", "пак", "по", "повече", "повечето", "под", "поне", "поради", "после", "почти", "прави", "пред",
				"преди", "през", "при", "пък", "първата", "първи", "първо", "пъти", "равен", "равна", "с", "са", "сам",
				"само", "се", "сега", "си", "син", "скоро", "след", "следващ", "сме", "смях", "според", "сред", "срещу",
				"сте", "съм", "със", "също", "т", "тази", "така", "такива", "такъв", "там", "твой", "те", "тези", "ти",
				"т.н.", "то", "това", "тогава", "този", "той", "толкова", "точно", "три", "трябва", "тук", "тъй", "тя",
				"тях", "у", "утре", "харесва", "хиляди", "ч", "часа", "че", "често", "чрез", "ще", "щом", "юмрук", "я",
				"як", "zhelyo", "не!новините", "\"дневник\"", "br2n");

		final CharArraySet stopSet = new CharArraySet(stopWords, false);
		BULGARIAN_STOP_WORDS_SET = CharArraySet.unmodifiableSet(stopSet);
	}

	public static void main(String[] args) throws IOException {
		System.out.println(NumberUtils.isDigits("12345"));
		System.out.println(NumberUtils.isDigits("12345.1"));
		System.out.println(NumberUtils.isDigits("12345,2"));
		
		System.out.println(NumberUtils.isNumber("12345"));
		System.out.println(NumberUtils.isNumber("12345.1"));
		System.out.println(NumberUtils.isNumber("12345,2".replace(",", ".")));
		System.out.println(NumberUtils.isNumber("12345,2"));
		StringReader input = new StringReader(
				"Правя тест на класификатор и после др.Дулитъл, пада.br2n ще се оправя с данните! които,са много зле. Но това е по-добре. Но24"
						.replaceAll("br2n", ""));

		LetterTokenizer tokenizer = new LetterTokenizer();
		tokenizer.setReader(input);

		TokenFilter stopFilter = new StopFilter(tokenizer, BULGARIAN_STOP_WORDS_SET);
		TokenFilter length = new LengthFilter(stopFilter, 3, 1000);
		TokenFilter stemmer = new BulgarianStemFilter(length);
		TokenFilter ngrams = new ShingleFilter(stemmer, 2, 2);

		try (TokenFilter filter = ngrams) {

			Attribute termAtt = filter.addAttribute(CharTermAttribute.class);
			filter.reset();
			while (filter.incrementToken()) {
				String word = termAtt.toString().replaceAll(",", "\\.").replaceAll("\n|\r", "");
				System.out.println(word);
			}
		}
	}

}
