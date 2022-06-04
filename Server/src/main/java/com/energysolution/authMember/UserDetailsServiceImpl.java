package com.energysolution.authMember;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import com.energysolution.dto.UserDTO;
import com.energysolution.dto.UserRoleDTO;
import com.energysolution.mapper.UserMapper;

import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@Service
public class UserDetailsServiceImpl implements UserDetailsService{
	//UserDetailsService : 유저의 정보를 가져오는 인터페이스
	@Autowired
	private UserMapper userMapper;

	@Override
	public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
		int exist = userMapper.existUserById(username);
		if(exist == 1)
			new UsernameNotFoundException("등록되지 않은 사용자");
		
		UserDTO user = userMapper.getUser(username);
		UserRoleDTO userRole = userMapper.getUserRole(username);
		
		
		return new UserDetailsImpl(userRole,user);
	}
	
}
