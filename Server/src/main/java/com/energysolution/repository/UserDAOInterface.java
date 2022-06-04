package com.energysolution.repository;

import java.util.HashMap;

import com.energysolution.dto.UserDTO;

public interface UserDAOInterface {
	//회원 저장
	public void insertUser(UserDTO user);
	
	//회원 수정 (비밀번호)
	public void updateUser(UserDTO uservo);
	
	//회원 검색
	public UserDTO selectUser(String UserId);
	
	//회원 삭제
	public void deleteUser(String UserId);
		
}
