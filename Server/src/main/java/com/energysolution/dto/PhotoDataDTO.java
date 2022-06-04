package com.energysolution.dto;

import com.google.gson.JsonObject;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class PhotoDataDTO {
	private String date;
	private String electUse;
	private String waterUse;
	private String electFee;
	private String waterFee;
	private String publicFee;
	private String totalFee;
	
	
	public JsonObject revertToJson() {
		JsonObject resultJSON = new JsonObject();
		resultJSON.addProperty("date", date);
		resultJSON.addProperty("electUse", electUse);
		resultJSON.addProperty("waterUse", waterUse);
		resultJSON.addProperty("electFee", electFee);
		resultJSON.addProperty("waterFee", waterFee);
		resultJSON.addProperty("publicFee", publicFee);
		resultJSON.addProperty("totalFee", totalFee);
		return resultJSON;
	}
}
