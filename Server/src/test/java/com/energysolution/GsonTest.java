package com.energysolution;

import org.junit.jupiter.api.Test;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

public class GsonTest {
	
	@Test
	void GsonTest() {
		Gson gson = new Gson();
		JsonObject object = new JsonObject();
		object.addProperty("name", "park");
		object.addProperty("age", 22);
		object.addProperty("success", true);
		String json = gson.toJson(object);
		System.out.println(json);
	}

}
