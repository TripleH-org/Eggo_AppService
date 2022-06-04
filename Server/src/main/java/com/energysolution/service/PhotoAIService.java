package com.energysolution.service;

import java.util.Map;

import com.energysolution.dto.PhotoDataDTO;

public interface PhotoAIService {
	// 고지서 사진 저장 & 데이터 가져오기
	public void insertPhoto(Map<String, Object> map);
	public PhotoDataDTO getPhotoData();	
	
	// 데이터 삭제
	public void deletePhotoData();
}
