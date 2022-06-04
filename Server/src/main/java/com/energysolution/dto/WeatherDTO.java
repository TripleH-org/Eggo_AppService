package com.energysolution.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class WeatherDTO {
	//기온
	private String T1H;
	//1시간 강수량
	//private String RN1;
	//습도
	private String REH;
	//풍속
	private String WSD;
}
