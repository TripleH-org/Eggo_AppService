package com.energysolution.repository;

import org.mybatis.spring.SqlSessionTemplate;

import com.energysolution.dto.BillDTO;

public class BillDAO implements BillDAOInterface {
	BillDTO bill;
	private final SqlSessionTemplate sqlSession = null;
	
	public BillDTO test() {
		return sqlSession.selectOne(null);
	}
	
	@Override
	public BillDTO insert(BillDTO bill)  {
		this.bill = bill;
		
		return this.bill;
	}
	@Override
	public BillDTO select() {
		return bill;
	}

}
