package com.energysolution.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class DetailBillDTO {
	
	//필요한 필드값
	private int BillId;
	private String publicFee;
	private String electUse;
	private String waterUse;
	private String electFee;
	private String waterFee;
	
	public DetailBillDTO(TotalBillDTO totalDTO) {
		BillId = 0;
		publicFee = totalDTO.getPublicFee();
		electUse = totalDTO.getElectUse();
		waterUse = totalDTO.getWaterUse();
		electFee = totalDTO.getElectFee();
		waterFee = totalDTO.getWaterFee();
	}
}
