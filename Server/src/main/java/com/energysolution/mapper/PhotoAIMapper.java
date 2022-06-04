package com.energysolution.mapper;

import java.util.Map;

import org.apache.ibatis.annotations.Mapper;

import com.energysolution.dto.PhotoDataDTO;

@Mapper
public interface PhotoAIMapper {
	public void insertPhoto(Map<String, Object> map);
	public PhotoDataDTO getPhotoData();
	
	public int getMaxFileKey();
	
	public void deletePhoto();
	public void deleteGojiseo();
}
