package com.energysolution.controller;

import java.io.UnsupportedEncodingException;
import java.sql.Blob;
import java.util.HashMap;
import java.util.Map;

import javax.mail.MessagingException;
import javax.mail.internet.MimeMessage;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Controller;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;
import org.springframework.web.util.UriComponents;
import org.springframework.web.util.UriComponentsBuilder;

import com.energysolution.service.TestService;
import com.energysolution.service.WeatherAIService;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

@Controller
public class TestController {
	
	@Autowired
	TestService testService;
	
	@Autowired
    private JavaMailSender javaMailSender;
	
	@Autowired
	WeatherAIService weatherAIService;
	
	private final String url = "http://localhost:5000";
    
	@GetMapping("")
	public @ResponseBody String Root() {		
		return "This is Root";
	}
	
	@PostMapping(value = "/scanPhotoTest", produces = "text/plain;charset=UTF-8")
	public @ResponseBody String getPhotoTest(@RequestParam("file") MultipartFile file) {
		try {
			System.out.println("========scanphoto test========");
			RestTemplate template = new RestTemplate();
			
			HttpHeaders headers = new HttpHeaders();
			headers.setContentType(MediaType.MULTIPART_FORM_DATA);
			
			MultiValueMap<String, Object> params = new LinkedMultiValueMap<>();
			params.add("file", file);
			UriComponents uri = UriComponentsBuilder.fromHttpUrl(url+"/getPhoto").build(false);
			String response = template.getForObject(uri.toString(), String.class, params);			
			
			System.out.println("========fin01========");
			//ResponseEntity<String> res = template.getForEntity(url + "/getPhoto", String.class);
			System.out.println(response);
			//ResponseEntity<String> response = template.postForObject(url+"/getPhoto", requestEntity, String.class);
			//System.out.println(response);
			return "success";
		}catch(Exception e) {
			e.printStackTrace();
			return "fail";
		}
	}
	
	// 실시간 요금 추정
	@GetMapping(value = "/realTimeTest", produces = "text/plain;charset=UTF-8")
	public @ResponseBody String getWeatherTest(@RequestParam("region1") String region1,
			@RequestParam("region2") String region2, @RequestParam("region3") String region3) {
		System.out.println("=====getWeather 요청=====");

		// 리턴 메세지
		Gson gson = new Gson();
		JsonObject resultJSON = new JsonObject();

		// nx, ny 좌표 데이터 요청 후 API 값 넣기
		RestTemplate template = new RestTemplate();
		String query = "region1="+region1+"&region2="+region2+"&region3="+region3;
		UriComponents uri = UriComponentsBuilder.fromHttpUrl(url+"/getAddress").query(query).build(false);
		String response = template.getForObject(uri.toString(), String.class);
			
		if (response.isEmpty() || response.equals("fail")) {
			resultJSON.addProperty("result", "fail");
			resultJSON.addProperty("message", "위치 데이터 로딩에 실패하였습니다.");
			return gson.toJson(resultJSON);

		} else {
			resultJSON.addProperty("result", "success");
			resultJSON.addProperty("message", "위치 데이터입니다.");
			String[] nxny = response.split(" ");
			resultJSON.addProperty("nx", nxny[0]);
			resultJSON.addProperty("ny", nxny[1]);
		}
		return gson.toJson(resultJSON);
	}
	
	// 메일 전송
    @GetMapping("/mail")
    public String index() throws MessagingException, UnsupportedEncodingException {

        String to = "gmlwn7245@naver.com";
        String from = "gmlwn@test.com";
        String subject = "test";

        StringBuilder body = new StringBuilder();
        body.append("<html> <body><h1>임시 비밀번호는 다음과 같습니다. </h1>");
        body.append("<div>테스트 입니다2. <img src=\"cid:flower.jpg\"> </div> </body></html>");

        MimeMessage message = javaMailSender.createMimeMessage();
        MimeMessageHelper mimeMessageHelper = new MimeMessageHelper(message, true, "UTF-8");

        mimeMessageHelper.setFrom(from,"EnergySolution_TempPW_manager");
        mimeMessageHelper.setTo(to);
        mimeMessageHelper.setSubject(subject);
        mimeMessageHelper.setText(body.toString(), true);

        javaMailSender.send(message);

        return "하이";
    }
	
	// Python과 요청 사용
	@GetMapping("/PythonConn")
	public @ResponseBody String testPythonConn() {
		System.out.println("test====> /PythonConn");
		
		String url = "http://localhost:5000";
		
		//RestTemplate 객체 생성
		RestTemplate template = new RestTemplate();		
		String response = template.getForObject(url+"/getTest", String.class);
		System.out.println(response);
		
		//response = template.getForObject(url+"/getWeather", String.class);
		//System.out.println(response);
		
		return response;
	}
	
	@PostMapping("/TestGetImage")
	public @ResponseBody Map<String, Object> getTestPhoto(@RequestParam("file") MultipartFile file) {
		System.out.println("getImage 요청");
		
		//Convert to Blob
		Map<String, Object> param = new HashMap<String, Object>();
		String fileName = file.getOriginalFilename();
		byte[] bytes;		
		
		Map<String, Object> resultMap = new HashMap<String, Object>();
		try {
			if (file != null) {
				//Convert to Blob
				bytes = file.getBytes();
				try {
					Blob blob = new javax.sql.rowset.serial.SerialBlob(bytes);
					param.put("file", blob);
					param.put("file_name", fileName);
					param.put("file_size", blob.length());
				}catch(Exception e) {
					e.printStackTrace();
				}
				
				
				//photoAIService.insertPhoto(file);
				System.out.println(file);
				resultMap.put("result", "success");
			} else {
				resultMap.put("result", "fail");
			}
		} catch (Exception e) {
			resultMap.put("result", "fail");
			e.printStackTrace();
		}
		return resultMap;
	}
	
	
}
