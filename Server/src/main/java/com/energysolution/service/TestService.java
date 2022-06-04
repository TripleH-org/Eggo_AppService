package com.energysolution.service;

import java.util.HashMap;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.energysolution.mapper.TestMapper;

@Service
public class TestService {
	@Autowired
	TestMapper testMapper;
	//이미지 Blob로 저장
	public void InsertImage(HashMap<String, Object> map){
		System.out.println("===insert image service===");
		testMapper.InsertImage(map);
	}
}
