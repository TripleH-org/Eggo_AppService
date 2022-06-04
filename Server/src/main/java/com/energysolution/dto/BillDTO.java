package com.energysolution.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class BillDTO {
	
	//필요한 필드값
	private int BillId;
	private String date;
	private String totalFee;
	
	public BillDTO(TotalBillDTO totalDTO) {
		BillId=0;
		date = totalDTO.getDate();
		totalFee = totalDTO.getTotalFee();
	}

}
