package com.energysolution.service;

import com.energysolution.dto.WeatherDTO;


public interface WeatherAIService {	
	// weather 정보 리턴
	public WeatherDTO returnWeatherData(String nx, String ny);
	
}
