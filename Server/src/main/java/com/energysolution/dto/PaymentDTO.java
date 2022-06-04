package com.energysolution.dto;


import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class PaymentDTO {
	private String UserId;
	private int BillId;

	public PaymentDTO(String UserId) {
		this.UserId = UserId;
	}
	

	@Override
	public String toString() {
		return "PaymentDTO [ UserId=" + UserId + ",BillId = " + BillId+"]";
	}
}
