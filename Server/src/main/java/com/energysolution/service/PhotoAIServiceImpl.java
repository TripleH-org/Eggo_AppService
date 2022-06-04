package com.energysolution.service;

import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.energysolution.dto.PhotoDataDTO;
import com.energysolution.mapper.PhotoAIMapper;

@Service
public class PhotoAIServiceImpl implements PhotoAIService{
	@Autowired
	PhotoAIMapper photoAIMapper;
	
	@Override
	public void insertPhoto(Map<String, Object> map) {
		System.out.println("=====Photo INSERT Service====");
		photoAIMapper.insertPhoto(map);
	}
	
	@Override
	public PhotoDataDTO getPhotoData() {
		System.out.println("=====Photo SELECT Service====");
		return photoAIMapper.getPhotoData();
	}

	@Override
	public void deletePhotoData() {
		photoAIMapper.deletePhoto();
		photoAIMapper.deleteGojiseo();
	}

	
	/*
	public String getMaxFileKey() {
		System.out.println("=====check MaxKeyValue Service====");
		int maxKey = photoAIMapper.getMaxFileKey();
		if(maxKey==0)
			return "1";
		return Integer.toString(maxKey+1);
	}*/

}
