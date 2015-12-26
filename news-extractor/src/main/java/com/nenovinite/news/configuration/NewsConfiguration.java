package com.nenovinite.news.configuration;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class NewsConfiguration {
	private static final String CREDIBLE_OPT = "c";
	private static final String UNRELIABLE_OPT = "u";
	private static final String VERBOSE_OPT = "v";

	private static final String CREDIBLE_OPT_LONG = "credible";
	private static final String UNRELIABLE_OPT_LONG = "unreliable";
	private static final String VERBOSE_OPT_LONG = "verbose";

	private final boolean verbose;
	private final String credibleDataset;
	private final String unreliableDataset;

	private Options getOptions() {
		Options options = new Options();

		Option verbose = Option.builder(VERBOSE_OPT).argName(VERBOSE_OPT).longOpt(VERBOSE_OPT_LONG).hasArg(false)
				.desc("be extra verbose").build();
		Option credible = Option.builder(CREDIBLE_OPT).argName(CREDIBLE_OPT).longOpt(CREDIBLE_OPT_LONG).hasArg().numberOfArgs(1)
				.required().desc("path to credible dataset").build();
		Option unreliable = Option.builder(UNRELIABLE_OPT).argName(UNRELIABLE_OPT).longOpt(UNRELIABLE_OPT_LONG).hasArg()
				.numberOfArgs(1).required().desc("path to unreliable dataset").build();

		options.addOption(verbose);
		options.addOption(credible);
		options.addOption(unreliable);

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
			this.verbose = line.hasOption(VERBOSE_OPT);
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

}
