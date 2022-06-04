package com.energysolution.mapper;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.ibatis.annotations.Mapper;

import com.energysolution.dto.UserDTO;
import com.energysolution.dto.UserRoleDTO;

@Mapper
public interface UserMapper {
	public void insertUser(UserDTO userDTO);
	public void insertRole(UserRoleDTO userRoleDTO);
	public void updateUser(HashMap<String, String> updateMap);
	public List<String> FindUserId(String Email);
	
	
	public UserDTO FindUserPW(HashMap<String, String> findUserPWMap);
	public String getUserPassword(String UserId);
	public void deleteUser(String UserId);
	
	public int checkUser(String UserId);
	public int existUserById(String UserId);
	public int checkUserByIdEmail(HashMap<String, String> UserMap);
	public UserDTO getUser(String UserId);
	public UserRoleDTO getUserRole(String UserId);
	
	// # 안드로이드 데이터 요청 테스트용
	public List<Map<String, Object>> getReqData() throws Exception;
}
