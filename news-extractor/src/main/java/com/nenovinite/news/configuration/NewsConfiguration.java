package com.nenovinite.news.configuration;

import java.security.InvalidParameterException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;

import com.nenovinite.news.models.ModelBase;
import com.nenovinite.news.models.ModelLogisticRegression;
import com.nenovinite.news.models.ModelNaiveBayes;
import com.nenovinite.news.models.ModelSVM;

public class NewsConfiguration {
	private static final String CREDIBLE_OPT = "c";
	private static final String UNRELIABLE_OPT = "u";
	private static final String VALIDATION_OPT = "va";
	private static final String VERBOSE_OPT = "v";
	private static final String MODEL_OPT = "m";
	private static final String OUTPUT_OPT = "o";

	private static final String CREDIBLE_OPT_LONG = "credible";
	private static final String UNRELIABLE_OPT_LONG = "unreliable";
	private static final String VALIDATION_OPT_LONG = "validation";
	private static final String VERBOSE_OPT_LONG = "verbose";
	private static final String MODEL_OPT_LONG = "model";
	private static final String OUTPUT_OPT_LONG = "output";

	private final boolean verbose;
	private final String credibleDataset;
	private final String unreliableDataset;
	private final String model;
	private final String validationDataset;
	private final String outputFolder;

	private Options getOptions() {
		Options options = new Options();

		Option verbose = Option.builder(VERBOSE_OPT).argName(VERBOSE_OPT).longOpt(VERBOSE_OPT_LONG).hasArg(false)
				.desc("be extra verbose").build();
		Option credible = Option.builder(CREDIBLE_OPT).argName(CREDIBLE_OPT).longOpt(CREDIBLE_OPT_LONG).hasArg()
				.numberOfArgs(1).required().desc("path to credible dataset").build();
		Option unreliable = Option.builder(UNRELIABLE_OPT).argName(UNRELIABLE_OPT).longOpt(UNRELIABLE_OPT_LONG).hasArg()
				.numberOfArgs(1).required().desc("path to unreliable dataset").build();
		Option validation = Option.builder(VALIDATION_OPT).argName(VALIDATION_OPT).longOpt(VALIDATION_OPT_LONG).hasArg()
				.numberOfArgs(1).required().desc("path to unreliable dataset").build();
		Option output = Option.builder(OUTPUT_OPT).argName(OUTPUT_OPT).longOpt(OUTPUT_OPT_LONG).hasArg().numberOfArgs(1)
				.required().desc("path to unreliable dataset").build();
		Option model = Option.builder(MODEL_OPT).argName(MODEL_OPT).longOpt(MODEL_OPT_LONG).hasArg().numberOfArgs(1)
				.required().desc("model name nb, lg, svm").build();

		options.addOption(verbose);
		options.addOption(credible);
		options.addOption(unreliable);
		options.addOption(validation);
		options.addOption(output);
		options.addOption(model);

		return options;
	}

	public NewsConfiguration(String[] args) throws ParseException {
		// create the parser
		CommandLineParser parser = new DefaultParser();
		try {
			// parse the command line arguments
			CommandLine line = parser.parse(this.getOptions(), args);
			this.credibleDataset = line.getOptionValue(CREDIBLE_OPT);
			this.unreliableDataset = line.getOptionValue(UNRELIABLE_OPT);
			this.validationDataset = line.getOptionValue(VALIDATION_OPT);
			this.outputFolder = line.getOptionValue(OUTPUT_OPT);
			this.verbose = line.hasOption(VERBOSE_OPT);
			this.model = line.getOptionValue(MODEL_OPT);
		} catch (ParseException exp) {
			// oops, something went wrong
			System.err.println("Parsing failed.  Reason: " + exp.getMessage());
			throw exp;
		}

	}

	public boolean isVerbose() {
		return verbose;
	}

	public String getCredibleDataset() {
		return credibleDataset;
	}

	public String getUnreliableDataset() {
		return unreliableDataset;
	}

	public String getValidationDataset() {
		return validationDataset;
	}

	public String getOutputFolder() {
		return outputFolder;
	}

	public ModelBase getModel(JavaRDD<LabeledPoint> training) {
		switch (model) {
		case "nb":
			return new ModelNaiveBayes(training);
		case "lg":
			return new ModelLogisticRegression(training);
		case "svm":
			return new ModelSVM(training);
		default:
			throw new InvalidParameterException(
					"Cannot load model for value: " + model + ". Please select one of: nb, lg or svm.");
		}
	}
}
