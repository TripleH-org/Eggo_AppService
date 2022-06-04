package com.energysolution.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class UserAuthDTO {
	public static final String USER = "USER";
	
	private String UserId;
	private String Authority;
	
	public UserAuthDTO(String Authority) {
		this.Authority = Authority;
	}
	
}
