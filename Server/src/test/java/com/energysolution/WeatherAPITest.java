package com.energysolution;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.junit.jupiter.api.Test;

import com.energysolution.dto.WeatherDTO;

public class WeatherAPITest {
	// 서비스 인증키입니다. 공공데이터포털에서 제공해준 인증키를 넣어주시면 됩니다. 
	private String serviceKey = "vtjq%2FAY95mysnfmzSx%2FBI0ijID8CYa7fdWqbcRMpeeIV3%2FApXziRmhiSjZ7z1SIhMWKBY%2BC3JAZ6axGiPmt4lQ%3D%3D";
	private String urlForm = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst";
	
	@Test
	void Test() {
		try{ /*요청정보입력 * 아래와 같은 정보들은 사용자 가이드를 확인하여 찾아주시면 됩니다. * 위도 경도는 엑셀파일 안에 있습니다. * */ //자신이 조회를 원하는 지역의 경도와 위도를 입력해주세요 
			//형식 지정
			SimpleDateFormat sdformat = new SimpleDateFormat("yyyyMMdd hhmm");
			Calendar c = Calendar.getInstance();
			Date date = new Date();
			c.setTime(date);
			//1시간전
			c.add(Calendar.HOUR,-2);
			String today = sdformat.format(c.getTime());
			System.out.println("today:"+today);
			String[] times = today.split(" ");
			StringBuilder query = new StringBuilder();
			query.append("?");
			query.append("serviceKey=").append(serviceKey);
			String nx = "56"; //경도
			String ny = "71"; //위도
			query.append("&nx=").append(nx).append("&ny=").append(ny);
			
			String numOfRows = "10";
			String pageNo = "1";
			query.append("&numOfRows=").append(numOfRows).append("&pageNo=").append(pageNo);
			
			String base_date = times[0]; // 자신이 조회하고싶은 날짜를 입력해주세요 
			String base_time = times[1]; //자신이 조회하고싶은 시간대를 입력해주세요 
			query.append("&base_date=").append(base_date).append("&base_time=").append(base_time);
			
			String type = "JSON";
			query.append("&dataType=").append(type);
			
			// 정보를 모아서 URL정보를 만들면됩니다.
			String urlStr = urlForm+ query.toString(); 
			URL url = new URL(urlStr); // 위 urlStr을 이용해서 URL 객체를 만들어줍니다. 
			HttpURLConnection conn =(HttpURLConnection)url.openConnection();
			conn.setRequestMethod("GET");
			
			int res = conn.getResponseCode();
			System.out.println(urlStr);
			System.out.println("res====>"+res);
			
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
			String result = sb.toString();
			System.out.println("결과: " + result);
			
			
			JSONParser jsonParser = new JSONParser();
			JSONObject jsonObj = (JSONObject) jsonParser.parse(result);
			JSONObject parse_response = (JSONObject) jsonObj.get("response");
			JSONObject parse_body = (JSONObject) parse_response.get("body");// response 로 부터 body 찾아오기
			JSONObject parse_items = (JSONObject) parse_body.get("items");// body 로 부터 items 받아오기
			JSONArray parse_item = (JSONArray) parse_items.get("item");
			Long totalCount = (Long) parse_body.get("totalCount");			

			
			System.out.println(totalCount);
			WeatherDTO weatherDTO = new WeatherDTO();
			for(int i=0; i<totalCount; i++) {
				JSONObject obj = (JSONObject)parse_item.get(i);
				String category = (String) obj.get("category");
				/*
				1시간 기온 TMP ℃
				1시간 강수량 PCP 범주 (1 mm)
				습도 REH %		
				풍속 WSD m/s
				*/
				switch(category) {
					case "T1H":{
						weatherDTO.setT1H(category);
						break;
					}
					case "REH":{
						weatherDTO.setREH(category);
						break;
					}
					case "WSD":{
						weatherDTO.setWSD(category);
						break;
					}
				}	
			}
			
			String Location = nx+" "+ny;
			
			HashMap<String, Object> weatherMap = new HashMap<>();
			weatherMap.put("Location", Location);

			weatherMap.put("weatherDTO",weatherDTO);
			
			
		}catch(Exception e){ 
			System.out.println(e.getMessage()); 
		} 
	}
}
