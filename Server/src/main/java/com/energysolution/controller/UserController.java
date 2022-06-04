package com.energysolution.controller;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.InetAddress;
import java.util.HashMap;
import java.util.List;

import javax.mail.MessagingException;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.ParseException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import com.energysolution.dto.UserDTO;
import com.energysolution.service.UserServiceImpl;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

@RestController
@RequestMapping("/main")
public class UserController {

	private String result;

	@Autowired
	UserServiceImpl userService;
	
	@RequestMapping("")
	public @ResponseBody String mainView() throws IOException, ParseException {
		System.out.println("=====Main=====");
		System.out.println(InetAddress.getLocalHost());
		return "This is Main";
	}

	// 로그인
	@PostMapping(value="login",produces="text/plain;charset=UTF-8")
	public String loginUser(UserDTO userDTO) {
		System.out.println("==========================================");
		System.out.println("로그인:"+userDTO.getUserId());
		Gson gson = new Gson();
		UserDTO getUserDTO = userService.loginUser(userDTO.getUserId(), userDTO.getPassword());
		System.out.println("userID:"+getUserDTO.getUserId());
		System.out.println(getUserDTO.getUserId().equals(userDTO.getUserId()));
		
		JsonObject resultJSON = new JsonObject();
		if (!getUserDTO.getUserId().equals(userDTO.getUserId())) {
			resultJSON.addProperty("result", "fail");
			resultJSON.addProperty("message", "잘못된 아이디 또는 비밀번호 입니다.");
			return gson.toJson(resultJSON);
		}
		resultJSON.addProperty("result", "success");
		resultJSON.addProperty("message", getUserDTO.getName() + "님 로그인 성공!");
		resultJSON.addProperty("email", getUserDTO.getEmail());
		resultJSON.addProperty("name", getUserDTO.getName());

		return gson.toJson(resultJSON);
	}

	// 회원가입
	@PostMapping(value="register",produces="text/plain;charset=UTF-8")
	public String registerUser(UserDTO userDTO) {	
		Gson gson = new Gson();
		System.out.println("==========================================");
		System.out.println("회원가입 시도");
		System.out.println(gson.toJson(userDTO));
		result = userService.registerUser(userDTO);

		
		JsonObject resultJSON = new JsonObject();
		
		if (result != "success") {
			resultJSON.addProperty("result", "fail");
			resultJSON.addProperty("message", "등록 실패");
		}else {
			resultJSON.addProperty("result", "success");
			resultJSON.addProperty("message", userDTO.getName() + "님 회원가입 성공");
		}
			
		System.out.println("회원가입 성공");
		return gson.toJson(resultJSON);
	}

	// 아이디 중복 확인
	@GetMapping(value="checkId",produces="text/plain;charset=UTF-8")
	public String checkId(@RequestParam("userId") String UserId) {
		int count = userService.checkUser(UserId);
		System.out.println("==========================================");
		System.out.println("아이디 중복 체크:"+UserId);
		
		Gson gson = new Gson();
		JsonObject resultJSON = new JsonObject();
		if (count != 0) {
			resultJSON.addProperty("result", "fail");
			resultJSON.addProperty("message", "이미 존재하는 아이디입니다.");
		} else {
			resultJSON.addProperty("result", "success");
			resultJSON.addProperty("message", "사용 가능한 아이디입니다.");
		}
		return gson.toJson(resultJSON);
	}



}