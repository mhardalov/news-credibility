package com.nenovinite.news;

import java.io.Serializable;

import scala.runtime.AbstractFunction1;

public abstract class JavaFunction1<T, R> extends AbstractFunction1<T, R> implements Serializable{

	@Override
	public abstract R apply(T arg0);
}
