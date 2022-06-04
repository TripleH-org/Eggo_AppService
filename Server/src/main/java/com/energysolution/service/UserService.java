package com.energysolution.service;

import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.mail.MessagingException;

import com.energysolution.dto.UserDTO;

public interface UserService {
	//회원 가입
	public String registerUser(UserDTO userDTO);
	
	//회원 로그인
	public UserDTO loginUser(String UserId, String Password);
	
	//회원 ID 찾기
	public List<String> FindUserId(String Email);
	
	//회원 PW 찾기
	public String FindUserPW(HashMap<String, String> findUserPWMap) throws MessagingException, UnsupportedEncodingException;
	
	//회원 수정 (비밀번호)
	public String updateUser(HashMap<String, String> updateMap);
		
	//회원 삭제
	public String deleteUser(String UserId, String Password);
	
	//회원 유무 체크
	public int checkUser(String UserId);
	
	//회원 유무 체크
	public int checkUserByIdEmail(HashMap<String, String> UserMap);

	//비밀번호 가져오기
	public String getUserPassword(String UserId);
	
	
	// # 안드로이드 데이터 요청 테스트용
	public List<Map<String, Object>> getReqData() throws Exception;
}
