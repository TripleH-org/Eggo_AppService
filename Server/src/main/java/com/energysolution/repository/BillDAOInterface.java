package com.energysolution.repository;

import com.energysolution.dto.BillDTO;

public interface BillDAOInterface {
	//고지서 저장
	public BillDTO insert(BillDTO bill);
		
	//고지서 검색
	public BillDTO select();
}
