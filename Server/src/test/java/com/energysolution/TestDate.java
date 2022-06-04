package com.energysolution;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import org.junit.jupiter.api.Test;

public class TestDate {
	
	@Test
	void Test() {
		SimpleDateFormat sdformat = new SimpleDateFormat("yyyyMM");
		Calendar c = Calendar.getInstance();
		Date date = new Date();
		c.setTime(date);
		c.add(Calendar.MONTH, -2);
		String thisMon = sdformat.format(c.getTime());
		System.out.println(thisMon);
		c.clear();
		c.setTime(date);
		c.add(Calendar.MONTH, -8);
		thisMon = sdformat.format(c.getTime());
		System.out.println(thisMon);
		
		for(int i=1; i<=6; i++) {
			c.clear();
			c.setTime(date);
			c.add(Calendar.MONTH, -i);
			String month = sdformat.format(c.getTime());
			System.out.println(month);
		}
	}
}
