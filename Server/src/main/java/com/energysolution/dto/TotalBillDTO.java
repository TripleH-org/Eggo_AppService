package com.energysolution.dto;

import com.google.gson.JsonObject;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.RequiredArgsConstructor;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class TotalBillDTO {
	
	//필요한 필드값
	private String UserId;
	private int BillId;
	private String date;
	private String totalFee;
	private String electUse;
	private String waterUse;
	private String electFee;
	private String waterFee;
	private String publicFee;
	
	//null 전용
	public TotalBillDTO(int BillId) {
		this.BillId = BillId;
	}
	
	//json 타입 전환
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
	
	//json 타입 전환 - pie
	public JsonObject revertToJsonPie() {
		JsonObject resultJSON = new JsonObject();
		resultJSON.addProperty("electFee", electFee);
		resultJSON.addProperty("waterFee", waterFee);
		resultJSON.addProperty("publicFee", publicFee);
		
		int tFee = Integer.parseInt(totalFee);
		int eFee = Integer.parseInt(electFee);
		int wFee = Integer.parseInt(waterFee);
		int pFee = Integer.parseInt(publicFee);
		int iFee = tFee-(eFee+wFee+pFee);
		
		resultJSON.addProperty("individualFee", Integer.toString(iFee));
		return resultJSON;
	}
}
