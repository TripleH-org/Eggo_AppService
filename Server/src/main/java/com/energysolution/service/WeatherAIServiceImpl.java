package com.energysolution.service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.springframework.stereotype.Service;

import com.energysolution.dto.WeatherDTO;

@Service
public class WeatherAIServiceImpl implements WeatherAIService{
	// 날씨 API KEY값
	private String serviceKey = "vtjq%2FAY95mysnfmzSx%2FBI0ijID8CYa7fdWqbcRMpeeIV3%2FApXziRmhiSjZ7z1SIhMWKBY%2BC3JAZ6axGiPmt4lQ%3D%3D";
	// 날씨 API 주소
	private String urlForm = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst";
	


	// 날씨 정보 리턴하기
	@Override
	public WeatherDTO returnWeatherData(String nx, String ny) {
		String[] times = getTodayDate().split(" ");
		String urlStr = urlForm + getQuery(times, nx, ny);
		String result = getWeatherConn(urlStr);
		
		return getWeatherData(result);
	}

	
	//오늘 날짜 가져오기
	public String getTodayDate() {
		//형식 지정
		SimpleDateFormat sdformat = new SimpleDateFormat("yyyyMMdd hhmm");
		Calendar c = Calendar.getInstance();
		Date date = new Date();
		c.setTime(date);
		//1시간전
		c.add(Calendar.HOUR,-1);
		String today = sdformat.format(c.getTime());
		return today;
	}
	
	
	// 쿼리 생성
	public String getQuery(String[] times, String nx, String ny) {
		
		StringBuilder query = new StringBuilder();
		query.append("?");
		query.append("serviceKey=").append(serviceKey);
		query.append("&nx=").append(nx).append("&ny=").append(ny);
		
		String numOfRows = "10";
		String pageNo = "1";
		query.append("&numOfRows=").append(numOfRows).append("&pageNo=").append(pageNo);
		
		String base_date = times[0]; // 자신이 조회하고싶은 날짜를 입력해주세요 
		String base_time = times[1]; //자신이 조회하고싶은 시간대를 입력해주세요 
		query.append("&base_date=").append(base_date).append("&base_time=").append(base_time);
		
		String type = "JSON";
		query.append("&dataType=").append(type);
		return query.toString();
	}
	
	// 날씨 정보 받아오기
	public String getWeatherConn(String urlStr) {
		String result = null;
		try {
			URL url = new URL(urlStr);
			HttpURLConnection conn =(HttpURLConnection)url.openConnection();
			conn.setRequestMethod("GET");
			conn.setConnectTimeout(10000);
			
			int res = conn.getResponseCode();
			//System.out.println(urlStr);
			//System.out.println("res====>"+res);
			
			BufferedReader rd;
			if (conn.getResponseCode() >= 200 && conn.getResponseCode() <= 300) {
				rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
			} else {
				rd = new BufferedReader(new InputStreamReader(conn.getErrorStream()));
			}
			
			StringBuilder sb = new StringBuilder();
			String line;
			while ((line = rd.readLine()) != null) {
				sb.append(line);
			}
			rd.close();
			conn.disconnect();
			result = sb.toString();
		} catch (Exception e) {
			e.printStackTrace();
		}  
		return result;
	}
	
	// Json에서 정보 추출
	public WeatherDTO getWeatherData(String result){
		JSONParser jsonParser = new JSONParser();
		WeatherDTO weatherDTO = new WeatherDTO();
		try {
			JSONObject jsonObj = (JSONObject) jsonParser.parse(result);
			JSONObject parse_response = (JSONObject) jsonObj.get("response");
			JSONObject parse_body = (JSONObject) parse_response.get("body");// response 로 부터 body 찾아오기
			JSONObject parse_items = (JSONObject) parse_body.get("items");// body 로 부터 items 받아오기
			JSONArray parse_item = (JSONArray) parse_items.get("item");
			Long totalCount = (Long) parse_body.get("totalCount");
			for(int i=0; i<totalCount; i++) {
				JSONObject obj = (JSONObject)parse_item.get(i);
				String category = (String) obj.get("category");
				
				/*
				1시간 기온 T1H ℃
				습도 REH %		
				풍속 WSD m/s
				*/
				switch(category) {
					case "T1H":{
						weatherDTO.setT1H((String)obj.get("obsrValue"));
						break;
					}
					case "REH":{
						weatherDTO.setREH((String)obj.get("obsrValue"));
						break;
					}
					case "WSD":{
						weatherDTO.setWSD((String)obj.get("obsrValue"));
						break;
					}
				}	
			}}catch(Exception e) {
			e.printStackTrace();
		}
		
		return weatherDTO;
	}	
}
