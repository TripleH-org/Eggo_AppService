package com.energysolution.mapper;

import java.util.HashMap;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface TestMapper {
	public void InsertImage(HashMap<String, Object> map);
}
