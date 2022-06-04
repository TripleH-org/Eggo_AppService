package com.energysolution.dto;

import com.energysolution.model.UserRole;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class UserRoleDTO {
	private String UserId;
	private int roleId;
	private UserRole role;
	
	public UserRoleDTO(String UserId) {
		this.UserId = UserId;
		role = UserRole.USER;
		
		// role = UserRole.USER;	
		// 회원가입시 기본 권한은 USER로 설정
	}
}
