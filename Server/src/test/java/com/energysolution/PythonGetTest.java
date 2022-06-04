package com.energysolution;


import org.junit.jupiter.api.Test;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponents;
import org.springframework.web.util.UriComponentsBuilder;

public class PythonGetTest {
	
	private final String url = "http://168.131.197.45:5001";
	@Test
	public void test() {
		RestTemplate template = new RestTemplate();
		String query = "region1=서울특별시&region2=종로구&region3=사직동";
		UriComponents uri = UriComponentsBuilder.fromHttpUrl(url+"/getAddress").query(query).build(false);
		String response = template.getForObject(uri.toString(), String.class);
		System.out.println(response);
	}
}
